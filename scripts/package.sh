#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE="${TENSORFLOW_LUA_ROCK_TREE:-$ROOT/dist/rock}"
rm -rf "$TREE"
mkdir -p "$TREE"
cd "$ROOT"
luarocks make --tree="$TREE" --deps-mode=none tensorflow-lua-scm-1.rockspec
echo "installed rock tree: $TREE"
