#!/usr/bin/env bash
# Download the Sherpa-ONNX streaming Zipformer model used by the live ASR engine.
# Places the int8 encoder/decoder/joiner + tokens into models/sherpa-streaming-zipformer-en/.
set -euo pipefail
cd "$(dirname "$0")/.."

DST="models/sherpa-streaming-zipformer-en"
NAME="sherpa-onnx-streaming-zipformer-en-2023-06-26"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NAME}.tar.bz2"

if [ -f "$DST/tokens.txt" ]; then
  echo "==> Sherpa model already present in $DST"
  exit 0
fi

mkdir -p "$DST" models/_sherpa_tmp
echo "==> Downloading $NAME ..."
curl -L --http1.1 -C - -o "models/_sherpa_tmp/model.tar.bz2" "$URL"
echo "==> Extracting ..."
tar xf "models/_sherpa_tmp/model.tar.bz2" -C "models/_sherpa_tmp"

SRC="models/_sherpa_tmp/$NAME"
cp "$SRC"/encoder-*int8.onnx "$DST"/
cp "$SRC"/decoder-*int8.onnx "$DST"/
cp "$SRC"/joiner-*int8.onnx  "$DST"/
cp "$SRC"/tokens.txt         "$DST"/
rm -rf models/_sherpa_tmp
echo "==> Done. Model in $DST"
ls -la "$DST"
