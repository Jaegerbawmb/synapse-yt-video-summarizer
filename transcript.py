"""
transcript.py:
  1. PRIMARY  → yt-dlp downloads audio → OpenAI Whisper transcribes locally (free, offline)
  2. FALLBACK → youtube-transcript-api v1.x fetches captions
"""

import os
import re
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger("yt_summarizer.transcript")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        log.info("Loading Whisper model: %s", WHISPER_MODEL_SIZE)
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        log.info("Whisper model loaded ✓")
    return _whisper_model


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _download_audio(video_url: str, out_dir: str) -> Optional[str]:
    """Download best audio stream via yt-dlp. Returns file path or None."""
    try:
        import yt_dlp
    except ImportError:
        log.warning("yt-dlp not installed")
        return None

    out_template = os.path.join(out_dir, "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        for f in Path(out_dir).iterdir():
            if f.suffix in {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".opus"}:
                log.info("Audio downloaded: %s (%.1f MB)", f.name, f.stat().st_size / 1e6)
                return str(f)
        log.warning("yt-dlp finished but no audio file found")
        return None
    except Exception as e:
        log.warning("yt-dlp download failed: %s", e)
        return None


def _transcribe_with_whisper(audio_path: str) -> Tuple[str, Optional[str]]:
    model = _load_whisper()
    log.info("Transcribing with Whisper …")
    result = model.transcribe(audio_path, fp16=False, verbose=False)
    text = result.get("text", "").strip()
    lang = result.get("language")
    log.info("Whisper done: %d chars, lang=%s", len(text), lang)
    return text, lang


def _fetch_youtube_captions(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch captions using youtube-transcript-api v1.x.

    Per the docs:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)           # returns FetchedTranscript
        for snippet in fetched: print(snippet.text)  # iterable of FetchedTranscriptSnippet
        fetched.language_code                        # e.g. "en"

    Tries English first, then falls back to listing all available transcripts
    and taking whichever language is available.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        log.warning("youtube-transcript-api not installed")
        return None, None

    api = YouTubeTranscriptApi()

    # ── Attempt 1: fetch English directly (fastest path) ─────────────────────
    for lang_pref in [["en"], ["en-US"], ["en-GB"]]:
        try:
            fetched = api.fetch(video_id, languages=lang_pref)
            text = _snippets_to_text(fetched)
            if text:
                log.info("Captions fetched (lang=%s): %d chars", fetched.language_code, len(text))
                return text, fetched.language_code
        except Exception:
            continue

    # ── Attempt 2: list all transcripts, pick best available ─────────────────
    try:
        transcript_list = api.list(video_id)

        # Priority: manual English → auto English → manual any → auto any
        candidates = [
            lambda: transcript_list.find_manually_created_transcript(["en", "en-US", "en-GB"]),
            lambda: transcript_list.find_generated_transcript(["en", "en-US", "en-GB"]),
            lambda: transcript_list.find_manually_created_transcript(
                [t.language_code for t in transcript_list]
            ),
            lambda: transcript_list.find_generated_transcript(
                [t.language_code for t in transcript_list]
            ),
        ]

        for candidate_fn in candidates:
            try:
                transcript = candidate_fn()
                fetched = transcript.fetch()
                text = _snippets_to_text(fetched)
                if text:
                    log.info(
                        "Captions via list (lang=%s, generated=%s): %d chars",
                        transcript.language_code, transcript.is_generated, len(text)
                    )
                    return text, transcript.language_code
            except Exception:
                continue

    except Exception as e:
        log.warning("api.list() failed: %s", e)

    log.warning("No usable captions found for video_id=%s", video_id)
    return None, None


def _snippets_to_text(fetched) -> str:
    """Join FetchedTranscriptSnippet.text fields into a clean string."""
    parts = [snippet.text for snippet in fetched]
    text = " ".join(parts)
    text = re.sub(r"<[^>]+>", " ", text)   # strip any HTML tags
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Public API ────────────────────────────────────────────────────────────────

def get_transcript(video_id: str, video_url: str) -> Tuple[str, str, Optional[str]]:
    """
    Returns (transcript_text, source_label, language_code).
    source_label: "whisper" | "youtube_api"
    """

    # ── PRIMARY: Whisper (only if ffmpeg is on PATH) ──────────────────────────
    if _ffmpeg_available():
        with tempfile.TemporaryDirectory(prefix="yt_audio_") as tmpdir:
            audio_path = _download_audio(video_url, tmpdir)
            if audio_path:
                try:
                    text, lang = _transcribe_with_whisper(audio_path)
                    if text and len(text.split()) >= 20:
                        return text, "whisper", lang
                    log.warning("Whisper returned near-empty text – trying fallback")
                except Exception as e:
                    log.warning("Whisper error: %s – trying fallback", e)
    else:
        log.info(
            "ffmpeg not on PATH — skipping Whisper, using YouTube captions.\n"
            "  To enable Whisper: winget install ffmpeg  (then restart terminal)"
        )

    # ── FALLBACK: YouTube Transcript API ─────────────────────────────────────
    text, lang = _fetch_youtube_captions(video_id)
    if text and len(text.split()) >= 20:
        return text, "youtube_api", lang

    raise RuntimeError(
        "Could not obtain a transcript for this video.\n"
        "Possible reasons:\n"
        "  • Captions are disabled or unavailable for this video\n"
        "  • Video is private, age-gated, or region-locked\n"
        "  • ffmpeg not installed so Whisper audio path is unavailable\n"
        "    → Fix: run  winget install ffmpeg  then restart your terminal"
    )