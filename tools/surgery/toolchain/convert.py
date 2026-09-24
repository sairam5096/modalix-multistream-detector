#!/usr/bin/env python3
"""Ultralytics -> SiMa Modalix compiled pack (bf16, calibration-free).

Contract:  convert.py <model> <output_dir> [--imgsz N]

  <model> may be:
    * an Ultralytics .pt   (auto-exported to ONNX at --imgsz first), or
    * an .onnx detector.

Pipeline:
  0. (.pt only) Ultralytics export -> ONNX at the requested input size.
  1. Graph surgery (model_surgery/): auto-identify the YOLO family, rewrite the
     detection head to the SiMa box-decoder contract, embed labels, emit boxdecoder.json.
  2. ModelSDK quantize + compile (bf16, target modalix) -> compiled model pack
     (`*_mpk.tar.gz`) loadable by the on-SOM Neat runtime.
  3. Emit to <output_dir>: the compiled pack (+ boxdecoder.json bundled at the tar
     root so it is self-describing) + boxdecoder.json sidecar + logs.json.

Input size (before surgery):
  --imgsz N  sets the model input to 1x3xNxN. For a .pt it is applied at export;
  for an .onnx it is applied by re-simplifying to that shape (best effort). Surgery
  then reads the shape from the ONNX. Default 640.

Runs on an x86 host (ModelSDK 2.1 + model_surgery). Nothing here runs on the SOM.
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

# model_surgery/ lives next to this file; override with env if needed.
MODEL_SURGERY = os.environ.get("MODEL_SURGERY_DIR", str(Path(__file__).resolve().parent))
QC = os.environ.get("QUANTIZE_COMPILE", str(Path(__file__).resolve().parent / "quantize_compile.py"))
DEFAULT_IMGSZ = int(os.environ.get("TOOLCHAIN_IMGSZ", "640"))


def _inject_into_pack(src_pack, dst_pack, boxdecoder_path) -> None:
    """Rewrite src_pack -> dst_pack adding boxdecoder.json (with labels) at the tar root."""
    import tarfile
    with tarfile.open(src_pack, "r:gz") as tin, tarfile.open(dst_pack, "w:gz") as tout:
        for m in tin.getmembers():
            tout.addfile(m, tin.extractfile(m) if m.isfile() else None)
        tout.add(str(boxdecoder_path), arcname="boxdecoder.json")


def _log(logs: list, stage: str, ok: bool, detail: str = "") -> None:
    logs.append({"stage": stage, "ok": ok, "detail": detail, "t": time.time()})
    print(f"[{'OK' if ok else 'FAIL'}] {stage} {detail}", flush=True)


def export_pt_to_onnx(pt_path, imgsz: int, out_dir: Path) -> Path:
    """Export an Ultralytics .pt to a static ONNX (raw head, no NMS) at imgsz.

    `pt_path` may be a local file or a bare Ultralytics name (e.g. "yolo26n.pt") that
    Ultralytics auto-downloads.
    """
    from ultralytics import YOLO
    pt_path = Path(pt_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(pt_path))
    print(f"[export] {pt_path.name} (task={model.task}) -> ONNX @ {imgsz}x{imgsz}", flush=True)
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, opset=17,
                                  dynamic=False, batch=1, nms=False, simplify=True,
                                  device="cpu"))
    dst = out_dir / onnx_path.name
    if onnx_path.resolve() != dst.resolve():
        shutil.copy2(onnx_path, dst)
    return dst


def reshape_onnx(onnx_path: Path, imgsz: int, out_dir: Path) -> Path:
    """Best-effort: force an ONNX input to 1x3ximgszximgsz before surgery.

    Returns the original path if it already matches or if re-simplification fails.
    """
    import onnx
    m = onnx.load(str(onnx_path))
    inp = m.graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) == 4 and dims[2] == imgsz and dims[3] == imgsz:
        return onnx_path                                        # already right size
    try:
        from onnxsim import simplify
        shape = {inp.name: [1, 3, imgsz, imgsz]}
        m2, ok = simplify(str(onnx_path), overwrite_input_shapes=shape, dynamic_input_shape=False)
        if not ok:
            raise RuntimeError("onnxsim validation failed")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{onnx_path.stem}_{imgsz}.onnx"
        onnx.save(m2, str(out))
        print(f"[reshape] {onnx_path.name} -> 1x3x{imgsz}x{imgsz}", flush=True)
        return out
    except Exception as e:                                       # keep native shape
        print(f"[reshape] WARNING: could not reshape {onnx_path.name} to {imgsz} "
              f"({e}); using its native input size.", flush=True)
        return onnx_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Ultralytics .pt/.onnx -> SiMa Modalix pack (bf16).")
    ap.add_argument("model", help="Ultralytics .pt or an .onnx detector")
    ap.add_argument("output_dir")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                    help=f"square input size set before surgery (default {DEFAULT_IMGSZ})")
    args = ap.parse_args()

    model_arg = args.model
    suffix = Path(model_arg).suffix.lower()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    surgery_dir = out_dir / "_surgery"
    logs: list = []

    # A .pt may be a local file OR a bare Ultralytics name (auto-downloaded), so only an
    # .onnx must exist up front.
    if suffix == ".onnx" and not Path(model_arg).exists():
        print(f"onnx not found: {model_arg}", file=sys.stderr)
        return 2

    # 0) .pt -> ONNX (at imgsz) ; .onnx -> optionally reshape to imgsz --------------
    if suffix == ".pt":
        try:
            onnx_path = export_pt_to_onnx(model_arg, args.imgsz, out_dir)
        except Exception as e:
            _log(logs, "export", False, str(e)[-500:])
            (out_dir / "logs.json").write_text(json.dumps(logs, indent=2))
            return 1
        _log(logs, "export", True, f"{onnx_path.name} @ {args.imgsz}")
    elif suffix == ".onnx":
        onnx_path = reshape_onnx(Path(model_arg).resolve(), args.imgsz, out_dir)
    else:
        print(f"unsupported model type: {suffix} (use .pt or .onnx)", file=sys.stderr)
        return 2

    # 1) graph surgery -> box-decoder contract + boxdecoder.json --------------------
    env = dict(os.environ, PYTHONPATH=MODEL_SURGERY)
    r = subprocess.run(
        [sys.executable, "-m", "model_surgery", str(onnx_path), "--outdir", str(surgery_dir)],
        env=env, capture_output=True, text=True)
    _log(logs, "surgery", r.returncode == 0, (r.stdout + r.stderr)[-500:])
    if r.returncode != 0:
        (out_dir / "logs.json").write_text(json.dumps(logs, indent=2))
        return 1
    surgered = next(surgery_dir.glob("*_surgery.onnx"))
    boxdecoder = surgery_dir / "boxdecoder.json"

    # 2) ModelSDK quantize + compile (bf16, modalix) -> compiled pack ---------------
    build_dir = out_dir / "_compile"
    r = subprocess.run(
        [sys.executable, QC,
         "--model_path", str(surgered), "--model_format", "onnx", "--device", "modalix",
         "--bf16-weights", "--bf16-activations",
         "--input_shapes", f"1,3,{args.imgsz},{args.imgsz}",
         "--build_dir", str(build_dir)],
        capture_output=True, text=True)
    _log(logs, "compile", r.returncode == 0, (r.stdout + r.stderr)[-500:])
    if r.returncode != 0:
        (out_dir / "logs.json").write_text(json.dumps(logs, indent=2))
        return 1

    # 3) collect artifacts: bundle labels-carrying boxdecoder.json INTO the pack ----
    mpk = next(build_dir.rglob("*_mpk.tar.gz"))
    out_mpk = out_dir / mpk.name
    _inject_into_pack(mpk, out_mpk, boxdecoder)
    shutil.copy2(boxdecoder, out_dir / "boxdecoder.json")
    _log(logs, "emit", True, f"{mpk.name} (labels bundled) + boxdecoder.json sidecar")
    (out_dir / "logs.json").write_text(json.dumps(logs, indent=2))
    print(f"\nModalix pack ready: {out_dir}/{mpk.name}\n"
          f"  run on the SOM   : soc/cpp/neat_detect (C++) or soc/python/neat_detect.py (Python) — Neat 0.3.0\n"
          f"  test on a device : soc/test_on_device.sh <board-ip> {out_dir}/{mpk.name} <image>\n"
          f"  config           : boxdecoder.json (decode_type/labels/W-H/topk/thresholds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
