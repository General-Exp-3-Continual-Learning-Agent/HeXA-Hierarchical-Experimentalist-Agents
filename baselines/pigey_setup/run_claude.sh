#!/bin/bash
set -euo pipefail
PIGEY_SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
source "$PIGEY_SETUP_DIR/activate.sh"
exec python "$PIGEY_SETUP_DIR/runner/run_claude.py" "$@"
