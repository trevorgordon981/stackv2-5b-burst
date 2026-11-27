#!/usr/bin/env bash
# ============================================================================
# Deprecated compatibility wrapper.
# Use `go.sh` as the single supported burst launcher.
# This wrapper defaults TOKENIZER to the bundled `./tokenizer` directory and
# delegates directly to `go.sh` so all fail-closed transfer, integrity, and
# CUDA wheel (cu128) guarantees apply identically.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
echo "NOTE: setup-on-rental.sh is deprecated; delegating to $HERE/go.sh" >&2
export TOKENIZER="${TOKENIZER:-$HERE/tokenizer}"
exec bash "$HERE/go.sh" "$@"
