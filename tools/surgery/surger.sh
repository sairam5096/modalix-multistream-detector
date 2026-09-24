#!/usr/bin/env bash
# surger.sh — one command: a YOLO(v6/v8/v9/v10/v11/26/x) ONNX (or .pt) in,
# a deployable SiMa Modalix pack out. It runs graph surgery -> ModelSDK
# quantize+compile -> a self-describing *_surgery_mpk.tar.gz, then verifies the
# pack with ../../models/identify_model.py.
#
# The customer only provides the model. Everything else is automatic.
#
# Usage:
#   ./surger.sh <model.onnx|model.pt> [--out DIR] [--calib DIR] [--int8|--bf16]
#               [--num-calib N] [--imgsz N] [--name NAME]
#
#   --int8   (default) 8-bit PTQ — needs calibration images (--calib); highest FPS.
#   --bf16   calibration-free, ~lossless; larger/slower than int8 but no calib set.
#   --calib  a folder of representative images for int8 calibration. Use your OWN
#            domain images for a custom model; falls back to the toolchain's
#            bundled COCO set if omitted.
#
# Requires the SiMa conversion toolchain image `sima-ultralytics-toolchain`
# (obtain/build it from the SiMa Modalix SDK — it bundles the ModelSDK compiler).
# Set TOOLCHAIN_IMAGE to override the image name.
set -euo pipefail

IMAGE="${TOOLCHAIN_IMAGE:-sima-ultralytics-toolchain}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDENTIFY="$HERE/../../models/identify_model.py"

MODEL=""; OUT=""; CALIB=""; PREC="int8"; NUMCALIB="50"; IMGSZ=""; NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out)       OUT="$2"; shift 2;;
    --calib)     CALIB="$2"; shift 2;;
    --int8)      PREC="int8"; shift;;
    --bf16)      PREC="bf16"; shift;;
    --num-calib) NUMCALIB="$2"; shift 2;;
    --imgsz)     IMGSZ="$2"; shift 2;;
    --name)      NAME="$2"; shift 2;;
    -h|--help)   sed -n '2,32p' "$0"; exit 0;;
    *)           MODEL="$1"; shift;;
  esac
done
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "error: give a model .onnx/.pt as the first argument" >&2; exit 2; }

MODEL="$(readlink -f "$MODEL")"
OUT="${OUT:-$(dirname "$MODEL")/surger_out}"; mkdir -p "$OUT"; OUT="$(readlink -f "$OUT")"

# Stage everything under one dir so a single -v mount covers model, calib and output.
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
cp "$MODEL" "$WORK/$(basename "$MODEL")"
ENVS=(-e "NUM_CALIB_SAMPLES=$NUMCALIB")
[ -n "$IMGSZ" ] && ENVS+=(-e "TOOLCHAIN_IMGSZ=$IMGSZ")
[ "$PREC" = "int8" ] && ENVS+=(-e "TOOLCHAIN_INT8=1")
if [ "$PREC" = "int8" ] && [ -n "$CALIB" ]; then
  cp -r "$CALIB" "$WORK/calib"
  ENVS+=(-e "TOOLCHAIN_CALIB_DIR=/run/calib")
fi
mkdir -p "$WORK/out"

echo ">> $PREC surgery+compile of $(basename "$MODEL") via $IMAGE ..."
docker run --rm -v "$WORK":/run "${ENVS[@]}" "$IMAGE" "/run/$(basename "$MODEL")" /run/out

# Collect the pack + sidecars.
PACK="$(ls "$WORK"/out/*_mpk.tar.gz 2>/dev/null | head -1 || true)"
[ -n "$PACK" ] || { echo "error: no pack produced — see logs:"; cat "$WORK"/out/logs.json 2>/dev/null; exit 1; }
cp "$WORK"/out/*_mpk.tar.gz "$WORK"/out/boxdecoder.json "$WORK"/out/logs.json "$OUT"/ 2>/dev/null || true
PACK="$OUT/$(basename "$PACK")"
echo ">> pack: $PACK"

# Verify + emit a ready-to-paste config line and a matching labels file.
if [ -f "$IDENTIFY" ]; then
  echo ">> verifying with identify_model.py ..."
  python3 "$IDENTIFY" "$PACK" ${NAME:+--name "$NAME"} --write-labels "$OUT/labels.txt" || {
    echo "WARNING: identify_model.py flagged the pack — inspect $OUT/logs.json" >&2; exit 1; }
fi
echo ">> done. Deploy $PACK with decode_type from boxdecoder.json + $OUT/labels.txt"
