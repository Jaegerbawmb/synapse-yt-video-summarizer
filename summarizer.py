"""
summarizer.py – Four summarization strategies
=============================================
  textrank_summarize  → extractive  (sumy TextRank)
  t5_summarize        → abstractive (google-t5/t5-small)
  bart_summarize      → abstractive (facebook/bart-large-cnn)
  pegasus_summarize   → abstractive (google/pegasus-xsum)

Uses AutoModelForSeq2SeqLM + AutoTokenizer directly — compatible with
transformers v4.50+ which removed the "summarization" pipeline task alias.
"""

import logging
from typing import Optional

import torch

log = logging.getLogger("yt_summarizer.summarizer")

# ── Model identifiers ─────────────────────────────────────────────────────────
T5_MODEL      = "google-t5/t5-small"
BART_MODEL    = "facebook/bart-large-cnn"
PEGASUS_MODEL = "google/pegasus-xsum"

# Max *input* tokens each model's encoder accepts
T5_MAX_INPUT      = 512
BART_MAX_INPUT    = 1024
PEGASUS_MAX_INPUT = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model cache: { model_name: (tokenizer, model) } ──────────────────────────
_models: dict = {}


def _load(model_name: str):
    """Lazy-load and cache a seq2seq model + tokenizer."""
    if model_name not in _models:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        log.info("Loading %s on %s …", model_name, DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(DEVICE)
        model.eval()
        _models[model_name] = (tokenizer, model)
        log.info("Loaded ✓  %s", model_name)
    return _models[model_name]


# ── Text chunking ─────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_words: int, overlap: int = 10) -> list[str]:
    """Split text into word-count chunks with a small overlap."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + max_words]))
        start += max_words - overlap
    return chunks


# ── Core seq2seq generation ───────────────────────────────────────────────────

def _generate(
    model_name: str,
    text: str,
    max_input_tokens: int,
    min_new_tokens: int,
    max_new_tokens: int,
) -> str:
    """
    Tokenize → encode → generate → decode.
    Handles chunking for texts longer than max_input_tokens.
    """
    tokenizer, model = _load(model_name)

    # Estimate word budget (words ≈ tokens for English; rough but safe)
    word_budget = int(max_input_tokens * 0.75)
    chunks = _chunk_text(text, max_words=word_budget)
    log.info("  %s: %d chunk(s)", model_name, len(chunks))

    summaries = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=max_input_tokens,
            truncation=True,
            padding=False,
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summaries.append(decoded.strip())

    if len(summaries) == 1:
        return summaries[0]

    # Second pass: merge chunk summaries into one
    merged = " ".join(summaries)
    merge_chunks = _chunk_text(merged, max_words=int(max_input_tokens * 0.75))
    final_parts = []
    for chunk in merge_chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=max_input_tokens,
            truncation=True,
        ).to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                min_new_tokens=30,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        final_parts.append(tokenizer.decode(output_ids[0], skip_special_tokens=True).strip())

    return " ".join(final_parts)


# ── 1. TextRank (Extractive) ──────────────────────────────────────────────────

def textrank_summarize(text: str, num_sentences: int = 5) -> str:
    """
    Graph-based extractive summarization via sumy's TextRankSummarizer.
    Selects the most 'central' sentences from the original transcript —
    zero hallucination, fully interpretable.
    """
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words

    LANGUAGE = "english"
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    sentences = summarizer(parser.document, num_sentences)
    result = " ".join(str(s) for s in sentences)
    log.info("  TextRank: %d sentences → %d chars", num_sentences, len(result))
    return result or "TextRank could not extract meaningful sentences."


# ── 2. T5 (Abstractive) ───────────────────────────────────────────────────────

def t5_summarize(text: str, max_new_tokens: int = 200) -> str:
    """
    google-t5/t5-small — text-to-text transfer transformer.
    T5 requires the task prefix 'summarize: ' prepended to input.
    """
    prefixed = "summarize: " + text
    result = _generate(
        T5_MODEL, prefixed,
        max_input_tokens=T5_MAX_INPUT,
        min_new_tokens=40,
        max_new_tokens=max_new_tokens,
    )
    log.info("  T5: %d chars", len(result))
    return result


# ── 3. BART (Abstractive) ─────────────────────────────────────────────────────

def bart_summarize(text: str, max_new_tokens: int = 300) -> str:
    """
    facebook/bart-large-cnn — denoising autoencoder fine-tuned on CNN/DailyMail.
    Best for fluent, news-style summaries. Handles the longest input (1024 tokens).
    """
    result = _generate(
        BART_MODEL, text,
        max_input_tokens=BART_MAX_INPUT,
        min_new_tokens=50,
        max_new_tokens=max_new_tokens,
    )
    log.info("  BART: %d chars", len(result))
    return result


# ── 4. PEGASUS (Abstractive) ──────────────────────────────────────────────────

def pegasus_summarize(text: str, max_new_tokens: int = 200) -> str:
    """
    google/pegasus-xsum — gap-sentence generation pre-training on XSum.
    Produces the most concise, highly abstractive summaries of the three.
    """
    result = _generate(
        PEGASUS_MODEL, text,
        max_input_tokens=PEGASUS_MAX_INPUT,
        min_new_tokens=30,
        max_new_tokens=max_new_tokens,
    )
    log.info("  PEGASUS: %d chars", len(result))
    return result