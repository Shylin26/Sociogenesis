#!/bin/bash

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           SOCIOGENESIS — Install Script                  ║"
echo "║  20 identical agents. One hard problem. One MacBook.     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── check python ──────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "  ERROR: python3 not found. Install from https://python.org"
  exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python $PY_VERSION detected"

# ── check pip ─────────────────────────────────────────────────
if ! command -v pip3 &>/dev/null; then
  echo "  ERROR: pip3 not found. Run: python3 -m ensurepip"
  exit 1
fi

# ── create artifacts dir ──────────────────────────────────────
mkdir -p artifacts/code artifacts/research artifacts/visual

# ── install dependencies ──────────────────────────────────────
echo ""
echo "  Installing dependencies..."
echo ""

pip3 install --break-system-packages --quiet \
  torch \
  numpy \
  faiss-cpu \
  scikit-learn \
  networkx \
  fastapi \
  uvicorn \
  websockets \
  reportlab \
  httpx 2>&1 | grep -v "^$" | grep -v "already satisfied" | head -30

echo ""
echo "  Dependencies installed."

# ── set env ───────────────────────────────────────────────────
export KMP_DUPLICATE_LIB_OK=TRUE

# ── check if env already in zshrc ────────────────────────────
if ! grep -q "KMP_DUPLICATE_LIB_OK" ~/.zshrc 2>/dev/null; then
  echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.zshrc
  echo "  Added KMP_DUPLICATE_LIB_OK=TRUE to ~/.zshrc"
fi

# ── done ──────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Installation complete.                                  ║"
echo "║                                                          ║"
echo "║  Run the demo:                                           ║"
echo "║    export KMP_DUPLICATE_LIB_OK=TRUE                     ║"
echo "║    python3 run_demo.py                                   ║"
echo "║                                                          ║"
echo "║  Or run the full 1000-tick society:                      ║"
echo "║    python3 week8_loop.py                                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
