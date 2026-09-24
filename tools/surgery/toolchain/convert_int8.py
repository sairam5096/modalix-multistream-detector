#!/usr/bin/env python3
"""Ultralytics -> SiMa Modalix compiled pack (INT8 variant).

Same pipeline as convert.py, but the ModelSDK step runs 8-bit INT8 PTQ instead of
bf16. Unlike bf16 (calibration-free), INT8 needs a representative calibration set:
real images are run through the model to collect per-tensor activation ranges
(calib_method=mse). Without a calib dir it falls back to DUMMY random data
(low accuracy).

Contract:  convert_int8.py <model.pt|model.onnx> <output_dir> [--imgsz N]
                           [--calib-dir DIR] [--num-calib N] [--calib-method mse]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import export_pt_to_onnx, reshape_onnx, _log  # noqa: E402

MODEL_SURGERY = os.environ.get("MODEL_SURGERY_DIR", str(Path(__file__).resolve().parent))
QC = os.environ.get("QUANTIZE_COMPILE", str(Path(__file__).resolve().parent / "quantize_compile.py"))
DEFAULT_IMGSZ = int(os.environ.get("TOOLCHAIN_IMGSZ", "640"))


def _default_calib_dir() -> str:
    """Default INT8 calibration set: the bundled `calib100/` (100 COCO val imgs). We ship it so int8
    never silently falls back to dummy-random calibration. Override with --calib-dir or $CALIB_DIR
    (use your own domain images for a custom model)."""
    if os.environ.get("CALIB_DIR"):
        return os.environ["CALIB_DIR"]
    here = Path(__file__).resolve().parent                       # .../src
    for cand in (here.parent / "calib100",                       # repo root (local run)
                 Path("/opt/toolchain/calib100")):               # in the Docker image
        if cand.is_dir():
            return str(cand)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Ultralytics .pt/.onnx -> SiMa Modalix pack (INT8).")
    ap.add_argument("model", help="Ultralytics .pt or an .onnx detector")
    ap.add_argument("output_dir")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                    help=f"square input size set before surgery (default {DEFAULT_IMGSZ})")
    ap.add_argument("--calib-dir", default=_default_calib_dir(),
                    help="representative images for INT8 calibration (default: the bundled calib100/; "
                         "use your OWN domain images for a custom model)")
    ap.add_argument("--num-calib", type=int, default=int(os.environ.get("NUM_CALIB_SAMPLES", "50")))
    ap.add_argument("--calib-method", default=os.environ.get("CALIB_METHOD", "min_max"))
    args = ap.parse_args()

    model_arg = args.model
    suffix = Path(model_arg).suffix.lower()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    surgery_dir = out_dir / "_surgery"
    logs: list = []
    if suffix == ".onnx" and not Path(model_arg).exists():
        print(f"onnx not found: {model_arg}", file=sys.stderr)
        return 2

    # 0) .pt -> ONNX @ imgsz (local file or bare Ultralytics name) ; .onnx -> reshape
    if suffix == ".pt":
        try:
            onnx_path = export_pt_to_onnx(model_arg, args.imgsz, out_dir)
        except Exception as e:
            _log(logs, "export", False, str(e)[-500:])
            (out_dir / "logs.json").write_text(json.dumps(logs, indent=2)); return 1
        _log(logs, "export", True, f"{onnx_path.name} @ {args.imgsz}")
    elif suffix == ".onnx":
        onnx_path = reshape_onnx(Path(model_arg).resolve(), args.imgsz, out_dir)
    else:
        print(f"unsupported model type: {suffix} (use .pt or .onnx)", file=sys.stderr)
        return 2

    # 1) graph surgery -----------------------------------------------------------
    env = dict(os.environ, PYTHONPATH=MODEL_SURGERY)
    r = subprocess.run(
        [sys.executable, "-m", "model_surgery", str(onnx_path), "--outdir", str(surgery_dir)],
        env=env, capture_output=True, text=True)
    _log(logs, "surgery", r.returncode == 0, (r.stdout + r.stderr)[-500:])
    if r.returncode != 0:
        (out_dir / "logs.json").write_text(json.dumps(logs, indent=2)); return 1
    surgered = next(surgery_dir.glob("*_surgery.onnx"))
    boxdecoder = surgery_dir / "boxdecoder.json"

    # 2) ModelSDK quantize + compile (INT8, modalix) -----------------------------
    build_dir = out_dir / "_compile"
    cmd = [sys.executable, QC,
           "--model_path", str(surgered), "--model_format", "onnx", "--device", "modalix",
           "--input_shapes", f"1,3,{args.imgsz},{args.imgsz}",
           "--calib_method", args.calib_method, "--requant_mode", "sima",
           "--build_dir", str(build_dir)]
    calib = Path(args.calib_dir).resolve() if args.calib_dir else None
    if calib and calib.is_dir():
        cmd += ["--real_data", "--dataset_images", str(calib), "--num_calib_samples", str(args.num_calib)]
    else:
        _log(logs, "calib", True, "no calib dir -> DUMMY random calibration (low accuracy)")
    r = subprocess.run(cmd, capture_output=True, text=True)
    _log(logs, "compile", r.returncode == 0, (r.stdout + r.stderr)[-500:])
    if r.returncode != 0:
        (out_dir / "logs.json").write_text(json.dumps(logs, indent=2)); return 1

    # 3) collect artifacts (bundle labels-carrying boxdecoder.json into the pack) -
    from convert import _inject_into_pack
    mpk = next(build_dir.rglob("*_mpk.tar.gz"))
    _inject_into_pack(mpk, out_dir / mpk.name, boxdecoder)
    shutil.copy2(boxdecoder, out_dir / "boxdecoder.json")
    _log(logs, "emit", True, f"{mpk.name} (labels bundled) + boxdecoder.json sidecar")
    (out_dir / "logs.json").write_text(json.dumps(logs, indent=2))
    print(f"\nModalix INT8 pack ready: {out_dir}/{mpk.name}\n"
          f"  run on the SOM   : soc/cpp/neat_detect (C++) or soc/python/neat_detect.py (Python) — Neat 0.3.0\n"
          f"  test on a device : soc/test_on_device.sh <board-ip> {out_dir}/{mpk.name} <image>\n"
          f"  config           : boxdecoder.json (decode_type/labels/W-H/topk/thresholds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
