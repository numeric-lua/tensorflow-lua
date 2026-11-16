#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${TENSORFLOW_VERSION:-2.18.0}"
ARCH="${TENSORFLOW_ARCH:-linux-x86_64}"
DEST="$ROOT/third_party/libtensorflow"
if [[ -f "$DEST/lib/libtensorflow.so" ]]; then
  echo "libtensorflow already present at $DEST"
  exit 0
fi
mkdir -p "$DEST"
URL="https://storage.googleapis.com/tensorflow/versions/${VERSION}/libtensorflow-cpu-${ARCH}.tar.gz"
echo "downloading $URL"
curl -L --fail --show-error -o "$DEST/libtensorflow.tar.gz" "$URL"
tar -xzf "$DEST/libtensorflow.tar.gz" -C "$DEST"
rm -f "$DEST/libtensorflow.tar.gz"
ln -sf libtensorflow.so."$VERSION" "$DEST/lib/libtensorflow.so"
ln -sf libtensorflow_framework.so."$VERSION" "$DEST/lib/libtensorflow_framework.so"
echo "libtensorflow extracted to $DEST"
