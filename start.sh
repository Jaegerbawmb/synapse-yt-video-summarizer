#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# SYNAPSE – YouTube Video Summarizer
# Start script: installs deps (if needed) and launches the FastAPI backend
# ──────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/backend"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'
CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'

banner() {
  echo -e "${CYN}${BLD}"
  echo "  ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗"
  echo "  ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝"
  echo "  ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗  "
  echo "  ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝  "
  echo "  ███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗"
  echo "  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝"
  echo -e "${RST}"
  echo -e "  ${YLW}YouTube Summarizer${RST} — TextRank · T5 · BART · PEGASUS"
  echo ""
}

banner

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo -e "${RED}✗ python3 not found. Please install Python 3.10+${RST}"
  exit 1
fi
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "  ${GRN}✓ Python ${PYVER}${RST}"

# ── Check ffmpeg (required by Whisper) ───────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
  echo -e "  ${GRN}✓ ffmpeg found${RST}"
else
  echo -e "  ${YLW}⚠  ffmpeg not found – Whisper audio decoding will fail${RST}"
  echo -e "     Ubuntu:  sudo apt install ffmpeg"
  echo -e "     macOS:   brew install ffmpeg"
  echo -e "     Windows: https://ffmpeg.org/download.html"
fi

# ── Install Python deps ───────────────────────────────────────────────────────
echo ""
echo -e "  ${CYN}Installing / verifying Python dependencies…${RST}"
pip install -r requirements.txt -q --disable-pip-version-check

# Download NLTK data silently
python3 - <<'PY'
import nltk, os
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
PY
echo -e "  ${GRN}✓ Dependencies ready${RST}"

# ── Model pre-warm (optional, speeds up first request) ───────────────────────
PREWARM="${PREWARM_MODELS:-0}"
if [[ "$PREWARM" == "1" ]]; then
  echo ""
  echo -e "  ${CYN}Pre-warming HuggingFace models (this may take a few minutes)…${RST}"
  python3 - <<'PY'
from summarizer import _get_pipeline, T5_MODEL, BART_MODEL, PEGASUS_MODEL
for m in [T5_MODEL, BART_MODEL, PEGASUS_MODEL]:
    print(f"  loading {m} …")
    _get_pipeline(m)
print("  ✓ All models loaded")
PY
fi

# ── Launch ────────────────────────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-true}"

echo ""
echo -e "  ${GRN}${BLD}Starting API server…${RST}"
echo -e "  ${YLW}Backend:  http://${HOST}:${PORT}${RST}"
echo -e "  ${YLW}Docs:     http://${HOST}:${PORT}/docs${RST}"
echo -e "  ${YLW}Frontend: open frontend/index.html in your browser${RST}"
echo ""
echo -e "  ${CYN}Env overrides:${RST}"
echo -e "    WHISPER_MODEL=tiny|base|small|medium|large  (default: base)"
echo -e "    PREWARM_MODELS=1                            (pre-load HF models)"
echo -e "    PORT=8000                                   (API port)"
echo ""

if [[ "$RELOAD" == "true" ]]; then
  RELOAD_FLAG="--reload"
else
  RELOAD_FLAG=""
fi

exec uvicorn main:app --host "$HOST" --port "$PORT" $RELOAD_FLAG
