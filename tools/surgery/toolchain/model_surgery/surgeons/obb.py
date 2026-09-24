"""
YOLOv8 / v11 (/v12) oriented-bounding-box (OBB) head surgery.

An Ultralytics `-obb` model is the ordinary v8 DFL *detection* base — box branch
`cv2.{i}` (64ch DFL) + class branch `cv3.{i}` (nc classes, DOTA=15) — with one
extra head per scale: `cv4.{i}`, a **single-channel** Conv emitting the *raw*
oriented-box angle (Ultralytics `ne == 1`). This is the exact shape of pose's
`cv4` extra head, so surgery mirrors `surgeons/pose.py`: reuse the v8 bbox+class
rewrite (`_rewrite_detection`) PLUS re-expose the three raw `cv4.{i}` convs as
`angle_{0,1,2}`. The angle is NOT decoded in-graph.

Contract — 9 tensors, strides 8/16/32, input HxW:
    bbox_{i}        (1, 4,  H/s, W/s)   decoded cx,cy,w,h (pixel space)
    class_prob_{i}  (1, nc, H/s, W/s)   sigmoid
    angle_{i}       (1, 1,  H/s, W/s)   RAW cv4 angle head (pre-sigmoid)

Angle decode (Ultralytics `OBB` head): theta = (sigmoid(angle) - 0.25) * pi,
theta in [-pi/4, 3pi/4).

IMPORTANT rotation note: the reused v8 rewrite lands `bbox_{i}` via the plain
axis-aligned `dist2bbox` decode (grid offset added to the UNROTATED (r-l)/2,
(b-t)/2 centre), whereas full OBB uses `dist2rbox`, which rotates that same centre
offset by theta *before* adding the anchor (w,h = l+r, t+b are unaffected). NO
information is lost: the per-cell anchor is deterministic ((0.5+col)*s,
(0.5+row)*s), so a decoder recovers the raw offset as (bbox.cx - anchor_x,
bbox.cy - anchor_y), rotates it by theta = (sigmoid(angle)-0.25)*pi, and re-adds
the anchor to obtain the exact `dist2rbox` centre. The contract is therefore
sufficient for exact oriented-box reconstruction, BUT the on-device SiMa OBB
decoder must perform this anchor-aware rotation rather than treat `bbox_{i}` as
the final centre. The exact box-decoder OBB behaviour could not be confirmed
host-side; flagged for the on-device decoder spec. Box `w,h` and the raw angle
are exact regardless.

Validated (Ultralytics 8.4.104) against yolov8n-obb (/model.22, 15 cls) and
yolo11n-obb (/model.23, 15 cls): 9 outputs, correct shapes, finite, angle_i
bit-identical to the raw cv4 conv.
"""

from __future__ import annotations

import logging

import onnx

from .. import contract, onnx_helpers as oh
from ..identify import YoloIdentity
from .base import SurgeonBase
# shared v8 detection-base helpers live in pose.py (obb sits on the identical
# Detect base as pose/seg); reuse them here to stay DRY.
from .pose import _rewrite_detection, _expose, _head_conv_depth
from .yolo26 import SurgeonYolo26  # no-DFL/one2one base for yolo26-obb

log = logging.getLogger("model_surgery.surgeon.obb")


class SurgeonOBB(SurgeonBase):
    name = "yolov8-obb"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        head, H, W = ident.head_prefix, ident.height, ident.width
        # angle-head depth == Ultralytics `ne` (1 for OBB)
        ne = _head_conv_depth(model, f"{head}/cv4.0/cv4.0.2/Conv")
        log.info("obb surgery on %s: %d angle channel(s)", head, ne)

        # 1) v8 detection base -> bbox_{i} + class_prob_{i}
        _rewrite_detection(model, ident)

        # 2) expose the 3 raw cv4 angle heads as angle_{i} (no decode)
        for i, s in enumerate(contract.STRIDES):
            _expose(model, oh.find_node(model, f"{head}/cv4.{i}/cv4.{i}.2/Conv"), f"angle_{i}")
            oh.add_output(model, f"angle_{i}", (1, ne, H // s, W // s))

        return model


class SurgeonOBBV11(SurgeonOBB):
    """v11-obb — structurally identical head at its own /model.N index; the
    head-driven surgery applies verbatim. Kept distinct so dispatch/metadata
    report `yolov11-obb`. (v12-obb routes through the v8-obb surgeon, as v12
    detect identifies as the v8 family.)"""
    name = "yolov11-obb"


class SurgeonYolo26Obb(SurgeonYolo26):
    """YOLO26 OBB surgery. yolo26 no-DFL/one2one Detect base + single-channel angle head
    `{o}cv4.{i}.2` (o=one2one_). Mirrors SurgeonYolo26Seg (angle instead of mask, no proto).
    Angle is NOT decoded in-graph; the customer's rotated-box post-proc applies
    theta=(sigmoid(angle)-0.25)*pi + rotated NMS. Contract: 9 tensors (bbox_/class_prob_/angle_ x3)."""

    name = "yolo26-obb"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        head = ident.head_prefix
        o = ident.one2one_prefix or "one2one_"
        H, W = ident.height, ident.width
        ne = _head_conv_depth(model, f"{head}/{o}cv4.0/{o}cv4.0.2/Conv")   # 1 (angle)
        log.info("yolo26-obb surgery on %s: %d angle channel(s) (no-DFL/one2one)", head, ne)
        super().do_surgery(model, ident)                                   # bbox + class
        for i, s2 in enumerate(contract.STRIDES):
            _expose(model, oh.find_node(model, f"{head}/{o}cv4.{i}/{o}cv4.{i}.2/Conv"), f"angle_{i}")
            oh.add_output(model, f"angle_{i}", (1, ne, H // s2, W // s2))
        return model
