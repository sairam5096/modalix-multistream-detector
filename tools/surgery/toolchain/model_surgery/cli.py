"""
model_surgery CLI.

    python -m model_surgery <model.onnx> [--outdir DIR] [--labels FILE]
                            [--variant yolov8] [--topk 25] [--dtype bfloat16]

Flow:  identify -> audit -> surgery -> embed labels/metadata -> save
       -> re-audit -> contract check -> ORT numeric sanity -> boxdecoder.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import onnx

from . import __version__, audit as audit_mod, contract
from . import onnx_helpers as oh
from .identify import identify
from .labels import load_labels
from .surgeons import get_surgeon

log = logging.getLogger("model_surgery")


def _ort_sanity(model_path: str, height: int, width: int, expected: list[str]) -> tuple[bool, str]:
    """Run the surgered model once and confirm the contract outputs are sane."""
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        return True, "onnxruntime not installed — skipped numeric sanity"

    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        # match the model's input dtype (fp16 exports reject a float32 feed)
        dt = {"tensor(float16)": np.float16, "tensor(double)": np.float64}.get(inp.type, np.float32)
        x = np.random.rand(1, 3, height, width).astype(dt)
        outs = sess.run(None, {inp.name: x})
        names = [o.name for o in sess.get_outputs()]
        by = dict(zip(names, outs))
        for n in expected:
            if n not in by:
                return False, f"output '{n}' absent at runtime"
            a = by[n]
            if not np.all(np.isfinite(a)):
                return False, f"output '{n}' has non-finite values"
        shapes = {n: tuple(by[n].shape) for n in expected}
        return True, f"ran OK; output shapes {shapes}"
    except Exception as exc:
        return False, f"ORT run failed: {exc}"


def run(args: argparse.Namespace) -> int:
    src = Path(args.model)
    if not src.exists():
        log.error("model not found: %s", src)
        return 1

    outdir = Path(args.outdir) if args.outdir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"{src.stem}_surgery"
    out_model = outdir / f"{stem}.onnx"
    out_json = outdir / "boxdecoder.json"

    # 1) identify
    model, ident = identify(src, simplify=not args.no_simplify)
    if args.variant:
        ident.family = args.variant.lower().strip()   # force family (e.g. yolov7, yolox)
    print(f"\n[identify] {ident.describe()}  -> surgeon '{ident.surgeon_key}'\n")

    # 2) pre-surgery audit
    pre = audit_mod.audit_model(model, dtype=args.dtype)
    print(f"[audit:pre ] {pre.summary()}")

    # 3) surgery
    surgeon_cls = get_surgeon(ident.surgeon_key)
    if surgeon_cls is None:
        log.error("no surgeon registered for '%s'", ident.surgeon_key)
        return 1
    try:
        model = surgeon_cls().do_surgery(model, ident)
    except NotImplementedError as exc:
        log.error("surgery unavailable: %s", exc)
        return 3

    # 4) labels + metadata embedded IN the model
    labels = load_labels(args.labels, ident.num_classes)
    contract.embed_metadata(model, ident, labels=labels, topk=args.topk)

    # 5) save
    oh.save_model(model, str(out_model), simplify=not args.no_simplify)

    # 6) re-audit the surgered graph
    post = audit_mod.audit_model(str(out_model), dtype=args.dtype)
    print(f"[audit:post] {post.summary()}")

    # 7) contract check
    saved = onnx.load(str(out_model))
    specs = contract.output_specs(ident)
    check = contract.validate_contract(saved, ident)
    if check.ok:
        print(f"[contract  ] OK — {len(specs)} outputs match {ident.decode_type} box-decoder spec")
    else:
        print("[contract  ] FAIL:")
        for p in check.problems:
            print(f"             - {p}")

    # 8) numeric sanity
    exp_names = [n for n, _ in specs]
    ok, msg = _ort_sanity(str(out_model), ident.height, ident.width, exp_names)
    print(f"[runtime   ] {'OK' if ok else 'FAIL'} — {msg}")

    # 9) sidecar
    contract.write_boxdecoder_json(out_json, ident, labels=labels, topk=args.topk)

    print("\n[outputs]")
    print(f"  model         : {out_model}")
    print(f"  boxdecoder    : {out_json}")
    print(f"  labels        : {len(labels)} embedded in model metadata (key 'labels' + 'formatted_labels')")
    all_ok = check.ok and ok and post.clean
    print(f"\n[result] {'SUCCESS' if all_ok else 'COMPLETED WITH WARNINGS'}")
    return 0 if check.ok and ok else 4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="model_surgery",
        description="Auto-identify a YOLO ONNX and rewrite it to the SiMa box-decoder contract.",
    )
    p.add_argument("model", help="path to the input YOLO .onnx")
    p.add_argument("--outdir", help="output directory (default: alongside input)")
    p.add_argument("--name", help="output model stem (default: <input>_surgery)")
    p.add_argument("--labels", help="labels file (.txt one-per-line or .json); default COCO80/synth")
    p.add_argument("--variant", help="force variant, e.g. yolov8 (overrides auto-ID)")
    p.add_argument("--topk", type=int, default=contract.DEFAULT_TOPK, help="max detections (box-decoder)")
    p.add_argument("--dtype", choices=["int8", "bfloat16", "any"], default="bfloat16",
                   help="op-support policy for the audit (Modalix -> bfloat16)")
    p.add_argument("--no-simplify", action="store_true", help="skip onnxsim (also implicit if not installed)")
    p.add_argument("--version", action="version", version=f"model_surgery {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return run(args)
    except Exception as exc:
        log.exception("surgery failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
