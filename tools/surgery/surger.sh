#!/usr/bin/env bash
# surger.sh — one command: a YOLO(v6/v8/v9/v10/v11/26/x) ONNX (or .pt) in,
# a deployable SiMa Modalix pack out. It runs graph surgery -> ModelSDK
# quantize+compile -> a self-describing *_surgery_mpk.tar.gz, then verifies the
# pack with ../../models/identify_model.py.
#
# The customer only provides the model. Everything else is automatic.
#
# It runs the vendored conversion scripts in ./toolchain/ directly — intended to
# be run INSIDE the SiMa `sima-neat` (Model Compiler) container, which provides
# the ModelSDK (`afe`) the compile step needs. No separate toolchain package or
# image is required. (Use --docker to run the prebuilt sima-ultralytics-toolchain
# image instead, if you have it.)
#
# Usage:
#   ./surger.sh <model.onnx|model.pt> [--out DIR] [--calib DIR] [--int8|--bf16]
#               [--num-calib N] [--imgsz N] [--name NAME] [--docker]
#
#   --int8   (default) 8-bit PTQ — needs calibration images (--calib); highest FPS.
#   --bf16   calibration-free, ~lossless; larger/slower than int8 but no calib set.
#   --calib  a folder of representative images for int8 calibration. Use your OWN
#            domain images for a custom model.
#   --docker run the prebuilt `sima-ultralytics-toolchain` image instead of the
#            vendored scripts (set TOOLCHAIN_IMAGE to override the image name).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TC="$HERE/toolchain"
IDENTIFY="$HERE/../../models/identify_model.py"
IMAGE="${TOOLCHAIN_IMAGE:-sima-ultralytics-toolchain}"

MODEL=""; OUT=""; CALIB=""; PREC="int8"; NUMCALIB="50"; IMGSZ=""; NAME=""; USE_DOCKER=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out)       OUT="$2"; shift 2;;
    --calib)     CALIB="$2"; shift 2;;
    --int8)      PREC="int8"; shift;;
    --bf16)      PREC="bf16"; shift;;
    --num-calib) NUMCALIB="$2"; shift 2;;
    --imgsz)     IMGSZ="$2"; shift 2;;
    --name)      NAME="$2"; shift 2;;
    --docker)    USE_DOCKER=1; shift;;
    -h|--help)   sed -n '2,34p' "$0"; exit 0;;
    *)           MODEL="$1"; shift;;
  esac
done
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "error: give a model .onnx/.pt as the first argument" >&2; exit 2; }
MODEL="$(readlink -f "$MODEL")"
OUT="${OUT:-$(dirname "$MODEL")/surger_out}"; mkdir -p "$OUT"; OUT="$(readlink -f "$OUT")"
[ "$PREC" = "int8" ] && CONV="convert_int8.py" || CONV="convert.py"

verify() {  # $1 = pack path
  [ -f "$IDENTIFY" ] || { echo ">> pack: $1 (identify_model.py not found — skipping verify)"; return 0; }
  echo ">> verifying with identify_model.py ..."
  python3 "$IDENTIFY" "$1" ${NAME:+--name "$NAME"} --write-labels "$OUT/labels.txt" \
    || { echo "WARNING: identify_model.py flagged the pack — inspect $OUT/logs.json" >&2; return 1; }
}

if [ "$USE_DOCKER" = "1" ]; then
  # ---- prebuilt-image path -------------------------------------------------
  WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
  cp "$MODEL" "$WORK/$(basename "$MODEL")"; mkdir -p "$WORK/out"
  ENVS=(-e "NUM_CALIB_SAMPLES=$NUMCALIB"); [ -n "$IMGSZ" ] && ENVS+=(-e "TOOLCHAIN_IMGSZ=$IMGSZ")
  [ "$PREC" = "int8" ] && ENVS+=(-e "TOOLCHAIN_INT8=1")
  if [ "$PREC" = "int8" ] && [ -n "$CALIB" ]; then cp -r "$CALIB" "$WORK/calib"; ENVS+=(-e "TOOLCHAIN_CALIB_DIR=/run/calib"); fi
  echo ">> $PREC surgery+compile of $(basename "$MODEL") via docker image $IMAGE ..."
  docker run --rm -v "$WORK":/run "${ENVS[@]}" "$IMAGE" "/run/$(basename "$MODEL")" /run/out
  cp "$WORK"/out/*_mpk.tar.gz "$WORK"/out/boxdecoder.json "$WORK"/out/logs.json "$OUT"/ 2>/dev/null || true
else
  # ---- in-container (sima-neat) path: run the vendored scripts directly -----
  command -v activate-model-compiler >/dev/null 2>&1 && \
    { source activate-model-compiler >/dev/null 2>&1 || eval "$(activate-model-compiler 2>/dev/null)" || true; }
  python3 -c "import afe" 2>/dev/null || {
    echo "error: ModelSDK ('afe') not importable in this environment." >&2
    echo "       Run this inside the sima-neat / Model Compiler container, or use --docker." >&2; exit 3; }
  ARGS=("$MODEL" "$OUT"); [ -n "$IMGSZ" ] && ARGS+=(--imgsz "$IMGSZ")
  if [ "$PREC" = "int8" ] && [ -n "$CALIB" ]; then ARGS+=(--calib-dir "$(readlink -f "$CALIB")" --num-calib "$NUMCALIB"); fi
  echo ">> $PREC surgery+compile of $(basename "$MODEL") via vendored toolchain ..."
  ( cd "$TC" && PYTHONPATH="$TC" python3 "$CONV" "${ARGS[@]}" )
fi

PACK="$(ls "$OUT"/*_mpk.tar.gz 2>/dev/null | head -1 || true)"
[ -n "$PACK" ] || { echo "error: no pack produced — see $OUT/logs.json"; cat "$OUT/logs.json" 2>/dev/null; exit 1; }
echo ">> pack: $PACK"
verify "$PACK"
echo ">> done. Deploy $PACK with decode_type from boxdecoder.json + $OUT/labels.txt"
