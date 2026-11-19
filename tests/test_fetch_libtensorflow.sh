#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/fetch_libtensorflow.sh"
mkdir -p "$ROOT/tests/.tmp"
WORK="$(mktemp -d "$ROOT/tests/.tmp/fetch.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "fetch script test failed: $*" >&2
  exit 1
}

expect_failure() {
  local version="$1" pattern="$2" output status
  set +e
  output="$(TENSORFLOW_VERSION="$version" TENSORFLOW_ARCH=linux-x86_64 \
    TENSORFLOW_DEST="$WORK/dest" "$SCRIPT" 2>&1)"
  status=$?
  set -e
  [[ $status -ne 0 ]] || fail "$version unexpectedly succeeded"
  [[ "$output" == *"$pattern"* ]] ||
    fail "$version did not report '$pattern': $output"
}

expect_failure 2.19.0 "2.18 was the last release"
expect_failure 2.20.0 "2.18 was the last release"
expect_failure 2.99.0 "not a pinned release"
[[ ! -e "$WORK/dest" ]] || fail "unsupported version created $WORK/dest"

set +e
output="$(TENSORFLOW_VERSION=2.19.0 TENSORFLOW_ARCH=linux-x86_64 \
  TENSORFLOW_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
  TENSORFLOW_DEST="$WORK/dest" "$SCRIPT" 2>&1)"
status=$?
set -e
[[ $status -ne 0 && "$output" == *"2.18 was the last release"* ]] ||
  fail "checksum override bypassed the unavailable-version check"

for version in 2.16.2 2.18.0 2.18.1; do
  output="$(TENSORFLOW_VERSION="$version" TENSORFLOW_ARCH=linux-x86_64 \
    TENSORFLOW_DEST="$WORK/dest" TENSORFLOW_DRY_RUN=1 "$SCRIPT")"
  [[ "$output" == *"version=$version"* ]] ||
    fail "dry run did not report version $version"
  [[ "$output" == *"versions/$version/libtensorflow-cpu-linux-x86_64.tar.gz"* ]] ||
    fail "dry run did not report the official URL for $version"
  [[ "$output" =~ sha256=[0-9a-f]{64} ]] ||
    fail "dry run did not report a pinned checksum for $version"
done

mkdir -p "$WORK/dest/lib"
: > "$WORK/dest/lib/libtensorflow.so.2.18.0"
set +e
output="$(TENSORFLOW_VERSION=2.16.2 TENSORFLOW_ARCH=linux-x86_64 \
  TENSORFLOW_DEST="$WORK/dest" "$SCRIPT" 2>&1)"
status=$?
set -e
[[ $status -ne 0 && "$output" == *"contains a different version"* ]] ||
  fail "a stale destination was not rejected"

echo "fetch script version checks passed"
