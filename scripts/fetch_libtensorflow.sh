#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${TENSORFLOW_VERSION:-2.18.0}"
DEST="${TENSORFLOW_DEST:-$ROOT/third_party/libtensorflow}"
SUPPORTED="2.16.2, 2.18.0, 2.18.1"

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

# TensorFlow 2.18 was the last release with official Linux x86-64
# libtensorflow archives.  Do not fall through to a generic URL for later
# versions: the download would only produce a confusing 404.
if [[ "$ARCH" == "linux-x86_64" ]]; then
  case "$VERSION" in
    2.19|2.19.*|2.20|2.20.*)
      cat >&2 <<EOF
libtensorflow ${VERSION} is not available for ${ARCH}.
Google did not publish official Linux x86-64 CPU archives for TensorFlow
2.19 or 2.20; 2.18 was the last release with Linux x86 libtensorflow packages.
Supported Linux x86-64 versions: ${SUPPORTED}.
EOF
      exit 1
      ;;
  esac
fi

# Checksums are pinned for the official Linux x86-64 CPU archives.  Other
# architectures or builds still work when TENSORFLOW_SHA256 is supplied.
if [[ -z "${TENSORFLOW_SHA256:-}" && "$ARCH" == "linux-x86_64" ]]; then
  case "$VERSION" in
    2.16.2)
      TENSORFLOW_SHA256="bd7d4ccb25ef2798076ebc88e9fdbeaa08ffea57996813403bba5c69d219c2ad"
      ;;
    2.18.0)
      TENSORFLOW_SHA256="605bfcb370c7e7ec981eabada880f60784e3de018395be95c95c3e5592c3d9a2"
      ;;
    2.18.1)
      TENSORFLOW_SHA256="b692795f3ad198c531b02aeb2bc8146568d24aaf6a5dbf5faa43907c4028fd73"
      ;;
  esac
fi

if [[ "$ARCH" == "linux-x86_64" && -z "${TENSORFLOW_SHA256:-}" ]]; then
  cat >&2 <<EOF
libtensorflow ${VERSION} for ${ARCH} is not a pinned release.
Supported Linux x86-64 versions: ${SUPPORTED}.
Set TENSORFLOW_SHA256 to explicitly verify another build.
EOF
  exit 1
fi

URL="https://storage.googleapis.com/tensorflow/versions/${VERSION}/libtensorflow-cpu-${ARCH}.tar.gz"

if [[ "${TENSORFLOW_DRY_RUN:-}" == "1" ]]; then
  printf 'version=%s\narch=%s\nurl=%s\nsha256=%s\n' \
    "$VERSION" "$ARCH" "$URL" "${TENSORFLOW_SHA256:-}"
  exit 0
fi

if [[ -f "$DEST/lib/libtensorflow.so.${VERSION}" ]]; then
  ln -sf "libtensorflow.so.${VERSION}" "$DEST/lib/libtensorflow.so"
  ln -sf "libtensorflow_framework.so.${VERSION}" "$DEST/lib/libtensorflow_framework.so"
  echo "libtensorflow ${VERSION} already present at $DEST"
  exit 0
fi
if compgen -G "$DEST/lib/libtensorflow.so.*" >/dev/null; then
  echo "libtensorflow destination $DEST contains a different version;" >&2
  echo "remove it before fetching ${VERSION}" >&2
  exit 1
fi

mkdir -p "$DEST"
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
