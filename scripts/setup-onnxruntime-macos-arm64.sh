#!/bin/zsh
set -euo pipefail

readonly ORT_VERSION="1.20.1"
readonly ARCHIVE="onnxruntime-osx-arm64-${ORT_VERSION}.tgz"
readonly URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ARCHIVE}"
readonly INSTALL_ROOT="${ONNXRUNTIME_INSTALL_ROOT:-$HOME/.local/onnxruntime}"
readonly TARGET_DIR="${INSTALL_ROOT}/onnxruntime-osx-arm64-${ORT_VERSION}"

mkdir -p "$INSTALL_ROOT"
if [[ ! -f "$TARGET_DIR/lib/libonnxruntime.dylib" ]]; then
  temp_archive="$(mktemp -t onnxruntime.XXXXXX.tgz)"
  trap 'rm -f "$temp_archive"' EXIT
  curl --fail --location --retry 3 --output "$temp_archive" "$URL"
  tar -xzf "$temp_archive" -C "$INSTALL_ROOT"
fi

print "ONNX Runtime installed at: $TARGET_DIR"
print "export ONNXRUNTIME_ROOT=\"$TARGET_DIR\""
