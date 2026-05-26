"""
YouTube Video Summarizer - FastAPI Backend
==========================================
Pipeline:
  1. Extract transcript via Whisper (primary) or youtube-transcript-api (fallback)
  2. Extractive summarization via TextRank (sumy)
  3. Abstractive summarization via T5 / BART / PEGASUS (HuggingFace)
"""

import os
import re
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transcript import get_transcript
from summarizer import (
    textrank_summarize,
    t5_summarize,
    bart_summarize,
    pegasus_summarize,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("yt_summarizer.main")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="YouTube Video Summarizer API",
    description="Extractive (TextRank) + Abstractive (T5 / BART / PEGASUS) summarization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    url: str
    extractive_sentences: int = 5          # TextRank: how many sentences to keep
    abstractive_max_tokens: int = 300      # token budget for T5/BART/PEGASUS
    models: list[str] = ["textrank", "t5", "bart", "pegasus"]


class TranscriptInfo(BaseModel):
    source: str          # "whisper" | "youtube_api"
    language: Optional[str]
    word_count: int
    char_count: int
    snippet: str         # first 300 chars


class ModelResult(BaseModel):
    model: str
    summary: str
    elapsed_sec: float


class SummarizeResponse(BaseModel):
    video_id: str
    transcript: TranscriptInfo
    results: list[ModelResult]
    total_elapsed_sec: float


# ── Helpers ───────────────────────────────────────────────────────────────────
_YT_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)([\w\-]{11})"
)


def extract_video_id(url: str) -> str:
    m = _YT_RE.search(url)
    if not m:
        raise HTTPException(status_code=400, detail=f"Cannot parse YouTube video ID from: {url}")
    return m.group(1)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "message": "YouTube Summarizer API is running"}


@app.get("/models")
def available_models():
    return {
        "extractive": ["textrank"],
        "abstractive": ["t5", "bart", "pegasus"],
        "description": {
            "textrank": "Graph-based extractive – selects the most central sentences",
            "t5": "google-t5/t5-small – encoder-decoder seq2seq",
            "bart": "facebook/bart-large-cnn – denoising autoencoder fine-tuned on CNN/DM",
            "pegasus": "google/pegasus-xsum – gap-sentence generation pre-training",
        },
    }


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    total_start = time.perf_counter()

    video_id = extract_video_id(req.url)
    log.info("Processing video_id=%s", video_id)

    # ── 1. Transcript ─────────────────────────────────────────────────────────
    transcript_text, transcript_source, transcript_lang = get_transcript(video_id, req.url)

    if not transcript_text or len(transcript_text.split()) < 30:
        raise HTTPException(
            status_code=422,
            detail="Transcript too short or empty – cannot summarize.",
        )

    t_info = TranscriptInfo(
        source=transcript_source,
        language=transcript_lang,
        word_count=len(transcript_text.split()),
        char_count=len(transcript_text),
        snippet=transcript_text[:300].strip() + "…",
    )
    log.info("Transcript ready: source=%s words=%d", transcript_source, t_info.word_count)

    # ── 2. Summarization ──────────────────────────────────────────────────────
    results: list[ModelResult] = []
    requested = {m.lower() for m in req.models}

    def run(name: str, fn, *args, **kwargs):
        if name not in requested:
            return
        log.info("Running %s …", name)
        t0 = time.perf_counter()
        summary = fn(*args, **kwargs)
        elapsed = round(time.perf_counter() - t0, 2)
        results.append(ModelResult(model=name, summary=summary, elapsed_sec=elapsed))
        log.info("%s done in %.2fs", name, elapsed)

    run("textrank", textrank_summarize,
        transcript_text, num_sentences=req.extractive_sentences)

    run("t5", t5_summarize,
        transcript_text, max_new_tokens=req.abstractive_max_tokens)

    run("bart", bart_summarize,
        transcript_text, max_new_tokens=req.abstractive_max_tokens)

    run("pegasus", pegasus_summarize,
        transcript_text, max_new_tokens=req.abstractive_max_tokens)

    total_elapsed = round(time.perf_counter() - total_start, 2)
    log.info("All done in %.2fs", total_elapsed)

    return SummarizeResponse(
        video_id=video_id,
        transcript=t_info,
        results=results,
        total_elapsed_sec=total_elapsed,
    )


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
