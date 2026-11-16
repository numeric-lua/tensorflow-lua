#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${TENSORFLOW_VERSION:-2.18.0}"
DEST="$ROOT/third_party/libtensorflow"

if [[ -z "${TENSORFLOW_ARCH:-}" ]]; then
  case "$(uname -s):$(uname -m)" in
    Linux:x86_64|Linux:amd64) TENSORFLOW_ARCH="linux-x86_64" ;;
    *)
      echo "no default libtensorflow build for $(uname -s) $(uname -m); set TENSORFLOW_ARCH" >&2
      exit 1
      ;;
  esac
fi
ARCH="$TENSORFLOW_ARCH"

if [[ -z "${TENSORFLOW_SHA256:-}" && "$VERSION" == "2.18.0" &&
    "$ARCH" == "linux-x86_64" ]]; then
  TENSORFLOW_SHA256="605bfcb370c7e7ec981eabada880f60784e3de018395be95c95c3e5592c3d9a2"
fi


if [[ -f "$DEST/lib/libtensorflow.so" ]]; then
  echo "libtensorflow already present at $DEST"
  exit 0
fi
mkdir -p "$DEST"
URL="https://storage.googleapis.com/tensorflow/versions/${VERSION}/libtensorflow-cpu-${ARCH}.tar.gz"
TMP="$DEST/libtensorflow.tar.gz"
echo "downloading $URL"
curl -L --fail --show-error -o "$TMP" "$URL"
if [[ -n "${TENSORFLOW_SHA256:-}" ]]; then
  echo "$TENSORFLOW_SHA256  $TMP" | sha256sum -c -
fi
tar -xzf "$TMP" -C "$DEST"
rm -f "$TMP"
if [[ ! -f "$DEST/lib/libtensorflow.so.${VERSION}" ]]; then
  echo "unexpected libtensorflow archive layout under $DEST" >&2
  exit 1
fi
ln -sf "libtensorflow.so.${VERSION}" "$DEST/lib/libtensorflow.so"
ln -sf "libtensorflow_framework.so.${VERSION}" "$DEST/lib/libtensorflow_framework.so"
echo "libtensorflow ${VERSION} extracted to $DEST"
