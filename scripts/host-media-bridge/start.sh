#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HOST_MEDIA_VENV:-$HOME/.cache/zeroclaw-host-media/venv}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ ! -x "$UV_BIN" ]]; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  else
    echo "error: uv not found. Install uv or set UV_BIN." >&2
    exit 1
  fi
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$UV_BIN" venv "$VENV_DIR" --python 3.11
fi

"$UV_BIN" pip install --python "$VENV_DIR/bin/python" -r "$SCRIPT_DIR/requirements.txt"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/server.py"
