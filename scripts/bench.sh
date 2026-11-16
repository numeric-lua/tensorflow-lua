#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${TENSORFLOW_LIB:-}" && -z "${TENSORFLOW_ROOT:-}" ]]; then
  if [[ -f "$ROOT/third_party/libtensorflow/lib/libtensorflow.so" ]]; then
    export TENSORFLOW_ROOT="$ROOT/third_party/libtensorflow"
  else
    echo "libtensorflow not found; run ./scripts/fetch_libtensorflow.sh" >&2
    exit 1
  fi
fi
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
cd "$ROOT"
exec luajit bench/benchmark.lua "$@"
