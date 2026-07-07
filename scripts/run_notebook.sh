#!/usr/bin/env bash
# Launch the garden-path marimo notebook with remote-GPU-friendly defaults.
# Usage: ./scripts/run_notebook.sh [--edit] [--host H] [--port P] [--headless|--no-headless]
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m gp_notebook.serve "$@"
