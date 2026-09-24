#!/usr/bin/env python3
"""Identify a compiled SiMa model pack and emit a ready-to-use detector config.

Reads the box-decoder metadata baked into a compiled `.tar.gz` (or an unpacked
directory) and reports the detector family (yolov6 / yolo26 / yolov8 / ...), the
number of classes, and the input geometry. It then prints a `models:` entry you
can paste into a shared-multi-model-detector config and, on request, writes a
matching labels file — so a YOLOv6 model with ANY class count (3-class, 80-class,
custom) drops into the app without guesswork.

Why this matters: the detector decodes boxes with `num_classes` taken from the
labels file. Point a 3-class model at an 80-class `coco_label.txt` and the decode
is wrong (or crashes). This tool reads the real class count from the pack and
generates a labels file that matches it.

Usage:
    identify_model.py <pack.tar.gz | dir> [--name NAME] [--min-score 0.30]
                      [--write-labels labels.txt] [--json]

Exit code 0 = a supported detector family was identified, 2 = not identified.
"""
import argparse
import json
import os
import sys
import tarfile
import tempfile

# decode_type values the shared-multi-model-detector accepts (parse_box_decode_type)
SUPPORTED = {
    "yolo26", "yolov26", "yolov8", "yolov5", "yolov6", "yolov7", "yolov9",
    "yolov10", "yolox",
}


def _find_boxdecoder(root):
    """Return the parsed box-decoder json from an unpacked pack dir, or None."""
    # prefer the canonical name, fall back to the stage-prefixed one
    for cand in ("boxdecoder.json", "0_boxdecoder.json"):
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            with open(p) as fh:
                return json.load(fh)
    # some layouts nest it one level down
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith("boxdecoder.json"):
                with open(os.path.join(dirpath, f)) as fh:
                    return json.load(fh)
    return None


def _mpk_name(root):
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith("_mpk.json"):
                try:
                    return json.load(open(os.path.join(dirpath, f))).get("name")
                except Exception:
                    return None
    return None


_FAMILY_TOKENS = [
    ("yolo26", "yolo26"), ("yolov26", "yolo26"), ("yolov10", "yolov10"),
    ("yolov9", "yolov9"), ("yolov8", "yolov8"), ("yolov7", "yolov7"),
    ("yolov6", "yolov6"), ("yolov5", "yolov5"), ("yolox", "yolox"),
]


def _infer_from_mpk(root):
    """Best-effort identity for packs with no boxdecoder.json (e.g. yolo26 'raw'
    packs): family from the pack name, class count from the largest output shape's
    last dimension. These decode via `decode_type` + a labels file in the app."""
    mpk = None
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith("_mpk.json"):
                mpk = json.load(open(os.path.join(dirpath, f)))
                break
        if mpk:
            break
    if mpk is None:
        return None
    name = (mpk.get("name") or "").lower()
    family = next((fam for tok, fam in _FAMILY_TOKENS if tok in name), None)
    # class count = last dim of the widest output shape (the class-prob head)
    classes = None
    for p in mpk.get("plugins", []):
        for shp in (p.get("config_params", {}).get("params", {})
                    .get("output_shapes") or []):
            if shp and shp[-1] not in (4,):
                classes = shp[-1] if classes is None else max(classes, shp[-1])
    return {
        "identified": family in SUPPORTED,
        "source": "mpk.json (inferred, no box-decode head baked in)",
        "family": family, "decode_type": family,
        "model_version": None, "num_classes": classes,
        "model_width": None, "model_height": None, "strides": None,
        "input_depth": None, "bboxes_format": None, "labels": None,
        "pack_name": mpk.get("name"),
        "note": ("no box-decode head in the pack; the detector decodes it with "
                 f"decode_type={family} and a labels file of {classes} entries"),
    }


def identify(root):
    """Inspect an unpacked pack directory; return an info dict."""
    bd = _find_boxdecoder(root)
    info = {"identified": False, "source": None}
    if bd is None:
        inferred = _infer_from_mpk(root)
        if inferred is not None:
            return inferred
        info["error"] = ("no boxdecoder.json and no *_mpk.json in the pack — "
                         "cannot identify this model")
        return info

    info["source"] = "boxdecoder.json"
    family = (bd.get("model_family") or bd.get("decode_type") or "").lower()
    decode_type = (bd.get("decode_type") or family or "").lower()
    depth = bd.get("input_depth") or []
    # class count: explicit field, else infer from the head (cls tensors = the
    # channel groups that are NOT the 4-wide bbox reg tensors)
    num_classes = bd.get("num_classes")
    if num_classes is None and depth:
        cls = [d for d in depth if d != 4]
        num_classes = cls[0] if cls else None

    info.update({
        "identified": decode_type in SUPPORTED,
        "family": family or None,
        "decode_type": decode_type or None,
        "model_version": bd.get("model_version"),
        "num_classes": num_classes,
        "model_width": bd.get("model_width"),
        "model_height": bd.get("model_height"),
        "strides": bd.get("strides"),
        "input_depth": depth or None,
        "bboxes_format": bd.get("bboxes_format"),
        "labels": bd.get("labels") or None,
        "pack_name": _mpk_name(root),
    })
    if decode_type and decode_type not in SUPPORTED:
        info["error"] = (f"decode_type '{decode_type}' is not one the detector "
                         f"supports ({', '.join(sorted(SUPPORTED))})")
    return info


def with_unpacked(pack_path):
    """Yield a directory for pack_path (unpacking a .tar.gz to a temp dir)."""
    if os.path.isdir(pack_path):
        return pack_path, None
    tmp = tempfile.mkdtemp(prefix="idmodel_")
    with tarfile.open(pack_path) as tf:
        # only extract the small json/yaml metadata, never the big .elf
        members = [m for m in tf.getmembers()
                   if m.name.endswith((".json", ".yaml"))]
        tf.extractall(tmp, members=members)
    return tmp, tmp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", help="model .tar.gz or unpacked directory")
    ap.add_argument("--name", help="model name for the emitted config entry")
    ap.add_argument("--min-score", type=float, default=0.30)
    ap.add_argument("--labels-path", default="labels.txt",
                    help="labels path to reference in the emitted config entry")
    ap.add_argument("--write-labels", metavar="FILE",
                    help="write the pack's embedded class names to FILE")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    root, cleanup = with_unpacked(args.pack)
    try:
        info = identify(root)
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(cleanup, ignore_errors=True)

    if args.json:
        print(json.dumps(info, indent=2))
        return 0 if info.get("identified") else 2

    if not info.get("identified"):
        print(f"NOT a directly-usable detector model: {info.get('error', 'unknown')}",
              file=sys.stderr)
        return 2

    name = args.name or info.get("pack_name") or "model"
    print(f"Identified: {info['family']} (decode_type={info['decode_type']}, "
          f"v{info.get('model_version')})")
    print(f"  classes    : {info['num_classes']}")
    print(f"  input      : {info['model_width']}x{info['model_height']}  "
          f"strides={info['strides']}  head_depth={info['input_depth']}")
    labels = info.get("labels") or []
    if labels:
        preview = ", ".join(labels[:6]) + (" ..." if len(labels) > 6 else "")
        print(f"  labels     : {len(labels)} embedded -> {preview}")

    if args.write_labels and labels:
        with open(args.write_labels, "w") as fh:
            fh.write("\n".join(labels) + "\n")
        print(f"  wrote {len(labels)} labels -> {args.write_labels}")

    print("\n# paste into a shared-multi-model-detector config under `models:`")
    print(f"  - {{name: {name}, path: ../models/{os.path.basename(args.pack)}, "
          f"decode_type: {info['decode_type']}, labels: {args.labels_path}, "
          f"min_score: {args.min_score}}}")
    if info["num_classes"] and labels and len(labels) != info["num_classes"]:
        print(f"\nNOTE: embedded label count ({len(labels)}) != num_classes "
              f"({info['num_classes']}); use --write-labels and verify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
