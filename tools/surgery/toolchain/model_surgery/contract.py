"""
The SiMa generic box-decoder input contract — the single source of truth for
what surgery must produce per family, plus helpers to embed metadata/labels in
the model and emit the `boxdecoder.json` sidecar the plugin post-processor reads.

Three contracts (per the latest `genericboxdecode` MODELS.md):
  anchor-free (yolov6/8/9/10/11/yolo26): 6 tensors
      bbox_{i}       (1, 4,  H/s, W/s)   decoded pixel (cx,cy,w,h)
      class_prob_{i} (1, nc, H/s, W/s)   post-Sigmoid            [class_is_prob]
  anchor-based (yolov5/yolov7): 3 tensors
      raw_{i} (1, na*(5+nc), H/s, W/s)   raw logits              [decode in plugin]
  yolox: 3 tensors
      raw_{i} (1, 4+1+nc, H/s, W/s)      [reg,obj,cls]           [decode in plugin]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import onnx

from . import __version__
from .labels import BBOX_FORMAT, formatted_label_string

if TYPE_CHECKING:
    from .identify import YoloIdentity

log = logging.getLogger("model_surgery.contract")

STRIDES = (8, 16, 32)
DEFAULT_TOPK = 25

BBOX_PREFIX = "bbox_"
CLASS_PREFIX = "class_prob_"
RAW_PREFIX = "raw_"
KPT_PREFIX = "kpt_"       # pose keypoint heads
MASK_PREFIX = "mask_"     # seg mask-coeff heads
PROTO_NAME = "proto"
MASK_CH = 32              # Ultralytics mask coeffs / proto channels
PROTO_STRIDE = 4         # proto map is at H/4 x W/4
ANGLE_PREFIX = "angle_"  # obb per-scale oriented-box angle heads
ANGLE_CH = 1             # Ultralytics obb `ne` (single angle channel)
CLASS_SCORES_NAME = "class_scores"   # classify single (1, nc) output

BBOX_FORMAT_DECODED = "cxcywh_pixel"   # anchor-free post-surgery bbox channels


EFFDET_STRIDES = (8, 16, 32, 64, 128)   # EfficientDet P3..P7 (5 levels)


def _is_pose(ident) -> bool:
    return ident.family.endswith("-pose")


def _is_seg(ident) -> bool:
    return ident.family.endswith("-seg")


def _is_obb(ident) -> bool:
    return ident.family.endswith("-obb")


def _is_classify(ident) -> bool:
    return ident.family.endswith("-cls")


def _is_depth(ident) -> bool:
    return ident.family == "depth" or ident.family.endswith("-depth")


def _is_effdet(ident) -> bool:
    return ident.family == "efficientdet"


def _is_centernet(ident) -> bool:
    return ident.family == "centernet"


def strides_for(ident) -> tuple:
    if _is_classify(ident):
        return ()                # whole-image classifier, no spatial strides
    if _is_effdet(ident):
        return EFFDET_STRIDES
    if _is_centernet(ident):
        return (4,)              # single scale, stride 4
    return STRIDES

# COCO anchor priors (P3/8, P4/16, P5/32) for v5/v7 — the plugin's defaults too.
COCO_ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],
    [[30, 61], [62, 45], [59, 119]],
    [[116, 90], [156, 198], [373, 326]],
]


def bbox_output_names() -> list[str]:
    return [f"{BBOX_PREFIX}{i}" for i in range(len(STRIDES))]


def class_output_names() -> list[str]:
    return [f"{CLASS_PREFIX}{i}" for i in range(len(STRIDES))]


def raw_output_names() -> list[str]:
    return [f"{RAW_PREFIX}{i}" for i in range(len(STRIDES))]


def box_decoder_outputs(H: int, W: int, num_classes: int) -> list[tuple[str, tuple]]:
    """Anchor-free 6-tensor spec (surgeons call this directly)."""
    out = [(f"{BBOX_PREFIX}{i}", (1, 4, H // s, W // s)) for i, s in enumerate(STRIDES)]
    out += [(f"{CLASS_PREFIX}{i}", (1, num_classes, H // s, W // s)) for i, s in enumerate(STRIDES)]
    return out


# --------------------------------------------------------------------------- #
# per-family contract specs
# --------------------------------------------------------------------------- #
def _bbox_depth(ident: "YoloIdentity") -> int:
    if ident.is_yolox:
        return 4 + 1 + ident.num_classes
    if not ident.anchor_free:                       # v5/v7
        return ident.num_anchors * (5 + ident.num_classes)
    return 4


def output_specs(ident: "YoloIdentity") -> list[tuple[str, tuple]]:
    """The [(name, shape)] the surgered graph must expose for this family."""
    H, W, nc = ident.height, ident.width, ident.num_classes
    if _is_classify(ident):                          # single (1, nc) score vector
        return [(CLASS_SCORES_NAME, (1, nc))]
    if _is_depth(ident):                             # single dense depth map (1,1,H,W)
        return [("depth", (1, 1, H, W))]
    if _is_effdet(ident):                            # 5 levels: cls_0..4 + box_0..4
        na = ident.num_anchors
        specs = [(f"cls_{i}", (1, na * nc, H // s, W // s)) for i, s in enumerate(EFFDET_STRIDES)]
        specs += [(f"box_{i}", (1, na * 4, H // s, W // s)) for i, s in enumerate(EFFDET_STRIDES)]
        return specs
    if _is_centernet(ident):                         # single scale, stride 4
        return [("heatmap", (1, nc, H // 4, W // 4)),
                ("wh", (1, 2, H // 4, W // 4)),
                ("reg", (1, 2, H // 4, W // 4))]
    if ident.anchor_free and not ident.is_yolox:
        specs = box_decoder_outputs(H, W, nc)
        if _is_pose(ident):                          # + 3 keypoint heads (9 total)
            specs += [(f"{KPT_PREFIX}{i}", (1, 3 * ident.num_kpts, H // s, W // s))
                      for i, s in enumerate(STRIDES)]
        elif _is_seg(ident):                         # + 3 mask heads + proto (10 total)
            specs += [(f"{MASK_PREFIX}{i}", (1, MASK_CH, H // s, W // s))
                      for i, s in enumerate(STRIDES)]
            specs.append((PROTO_NAME, (1, MASK_CH, H // PROTO_STRIDE, W // PROTO_STRIDE)))
        elif _is_obb(ident):                         # + 3 angle heads (9 total)
            specs += [(f"{ANGLE_PREFIX}{i}", (1, ANGLE_CH, H // s, W // s))
                      for i, s in enumerate(STRIDES)]
        return specs
    d = _bbox_depth(ident)                            # v5/v7/yolox raw heads
    return [(f"{RAW_PREFIX}{i}", (1, d, H // s, W // s)) for i, s in enumerate(STRIDES)]


def input_depth(ident: "YoloIdentity") -> list[int]:
    if _is_classify(ident):
        return [ident.num_classes]                   # single score vector
    if _is_effdet(ident):
        na, nc = ident.num_anchors, ident.num_classes
        return [na * nc] * 5 + [na * 4] * 5
    if _is_centernet(ident):
        return [ident.num_classes, 2, 2]             # heatmap, wh, reg
    if ident.anchor_free and not ident.is_yolox:
        nc = ident.num_classes
        depths = [4, 4, 4, nc, nc, nc]
        if _is_pose(ident):
            depths += [3 * ident.num_kpts] * 3
        elif _is_seg(ident):
            depths += [MASK_CH] * 3 + [MASK_CH]      # mask heads + proto
        elif _is_obb(ident):
            depths += [ANGLE_CH] * 3                 # angle heads
        return depths
    d = _bbox_depth(ident)
    return [d, d, d]


def class_is_prob(ident: "YoloIdentity") -> bool:
    # v5/v7 raw heads are logits; anchor-free surgery bakes Sigmoid; yolox exports
    # are typically already sigmoided.
    return ident.anchor_free


# --------------------------------------------------------------------------- #
# metadata embedded IN the model (labels travel with the model)
# --------------------------------------------------------------------------- #
def embed_metadata(model: onnx.ModelProto, ident: "YoloIdentity", *,
                   labels: list[str], topk: int) -> None:
    # decoded-bbox families (anchor-free detect/pose/seg/obb) but NOT classify,
    # which carries no boxes.
    af = ident.anchor_free and not ident.is_yolox and not _is_classify(ident)
    payload = {
        "sima_tool": "nx_neat/model_surgery",
        "sima_tool_version": __version__,
        "decode_type": ident.decode_type,
        "task": ident.task,
        "bboxes_format": BBOX_FORMAT,
        "class_is_prob": "true" if class_is_prob(ident) else "false",
        "model_family": ident.family,
        "model_version": str(ident.version),
        "model_flavor": ident.flavor or "",
        "input_width": str(ident.width),
        "input_height": str(ident.height),
        "num_classes": str(ident.num_classes),
        "strides": json.dumps(list(strides_for(ident))),
        "input_depth": json.dumps(input_depth(ident)),
        "num_in_tensor": str(len(output_specs(ident))),
        "topk": str(topk),
        "output_names": json.dumps([n for n, _ in output_specs(ident)]),
        "labels": json.dumps(labels),
        "formatted_labels": formatted_label_string(labels),
    }
    if af:
        payload["bbox_format"] = BBOX_FORMAT_DECODED
    if not ident.anchor_free:                       # v5/v7
        payload["num_anchors"] = str(ident.num_anchors)
        payload["anchors"] = json.dumps(ident.anchors or COCO_ANCHORS)
    if _is_pose(ident):
        payload["num_keypoints"] = str(ident.num_kpts)
    if _is_seg(ident):
        payload["mask_channels"] = str(MASK_CH)
        payload["proto_stride"] = str(PROTO_STRIDE)
    if _is_obb(ident):
        payload["angle_channels"] = str(ANGLE_CH)

    keep = [kv for kv in model.metadata_props if kv.key not in payload]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    for k, v in payload.items():
        e = model.metadata_props.add(); e.key = k; e.value = v
    log.info("embedded %d metadata keys (incl. %d labels) into the model", len(payload), len(labels))


# --------------------------------------------------------------------------- #
# boxdecoder.json sidecar (the plugin post-processor's config)
# --------------------------------------------------------------------------- #
def write_boxdecoder_json(path, ident: "YoloIdentity", *, labels: list[str], topk: int) -> None:
    af = ident.anchor_free and not ident.is_yolox and not _is_classify(ident)
    H, W = ident.height, ident.width
    specs = output_specs(ident)
    cfg = {
        "decode_type": ident.decode_type,
        "task": ident.task,
        "bboxes_format": BBOX_FORMAT,
        "class_is_prob": class_is_prob(ident),
        "model_width": W, "model_height": H,
        "original_width": W, "original_height": H,
        "num_classes": ident.num_classes,
        "topk": topk,
        "detection_threshold": 0.5,
        "nms_iou_threshold": 0.3,
        "strides": list(strides_for(ident)),
        "num_in_tensor": len(specs),
        "input_depth": input_depth(ident),
        # spatial feature-map size per output tensor; classify's rank-2
        # (1, nc) vector has none, so fall back to the model input size.
        "input_height": [shp[2] if len(shp) > 2 else H for _, shp in specs],
        "input_width": [shp[3] if len(shp) > 3 else W for _, shp in specs],
        "output_names": [n for n, _ in specs],
        "model_version": str(ident.version),
        "model_family": ident.family,
        "labels": labels,
    }
    if af:
        cfg["bbox_format"] = BBOX_FORMAT_DECODED
    if not ident.anchor_free:                       # v5/v7 anchor-based
        cfg["num_anchors"] = ident.num_anchors
        # the ACTUAL anchor priors read from the graph (differ for v5 vs v7); COCO v5 as fallback
        cfg["anchors"] = ident.anchors or COCO_ANCHORS
    if _is_pose(ident):
        cfg["num_keypoints"] = ident.num_kpts
    if _is_seg(ident):
        cfg["mask_channels"] = MASK_CH
        cfg["proto_stride"] = PROTO_STRIDE
    if _is_obb(ident):
        cfg["angle_channels"] = ANGLE_CH
    Path(path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    log.info("wrote boxdecoder.json (%s) -> %s", ident.decode_type, path)


# --------------------------------------------------------------------------- #
# validator — did surgery produce this family's contract?
# --------------------------------------------------------------------------- #
@dataclass
class ContractCheck:
    ok: bool
    problems: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def validate_contract(model: onnx.ModelProto, ident: "YoloIdentity") -> ContractCheck:
    expected = dict(output_specs(ident))
    got = {o.name: o for o in model.graph.output}
    problems: list[str] = []
    for name, shape in expected.items():
        if name not in got:
            problems.append(f"missing output '{name}'"); continue
        dims = tuple(d.dim_value if (d.HasField("dim_value") and d.dim_value > 0) else None
                     for d in got[name].type.tensor_type.shape.dim)
        if len(dims) != len(shape):
            problems.append(f"'{name}' rank {len(dims)} != {len(shape)}")
        else:
            for ax, (g, e) in enumerate(zip(dims, shape)):
                if g is not None and g != e:
                    problems.append(f"'{name}' dim[{ax}]={g} != {e}")
    extra = set(got) - set(expected)
    if extra:
        problems.append(f"unexpected extra outputs: {sorted(extra)}")
    return ContractCheck(ok=not problems, problems=problems)
