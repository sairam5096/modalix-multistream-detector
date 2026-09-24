"""
YOLOv8 / v9 / v10 / v11 instance-segmentation head surgery.

An Ultralytics -seg model is the ordinary v8 DFL *detection* base — box branch
`cv2.{i}` (64ch DFL) + class branch `cv3.{i}` (nc classes) — with two extras:
per scale a `cv4.{i}` Conv emitting 32 raw *mask coefficients*, and a single
`proto` branch (`.../proto/...` ending in a Conv+SiLU) producing the
(1, 32, H/4, W/4) prototype masks. The SiMa generic box-decoder
(`decode_type=yolov8-seg`) assembles the masks itself, so surgery is exactly the
v8 bbox+class rewrite (reused from `SurgeonYoloV8` via `_rewrite_detection`) PLUS
re-exposing the three raw `cv4.{i}` convs as `mask_{0,1,2}` and the proto branch
as `proto`. Neither extra head is decoded in-graph.

Contract — 10 tensors, strides 8/16/32, input HxW:
    bbox_{i}        (1, 4,   H/s, W/s)   decoded cx,cy,w,h (pixel space)
    class_prob_{i}  (1, nc,  H/s, W/s)   sigmoid
    mask_{i}        (1, 32,  H/s, W/s)   RAW cv4 mask-coefficient head
    proto           (1, 32,  H/4, W/4)   prototype masks (160x160 @ 640 input)

Box-decoder mask assembly, per surviving detection m (32 coeffs c_m gathered
from the mask_{i} tensor at that detection's grid cell):
    M = sigmoid( sum_k c_m[k] * proto[k] )   -> (H/4, W/4) logits then prob
    M = crop_to_box(M, box)  ->  upsample(M, HxW)  ->  M > 0.5

Validated (Ultralytics 8.4.95) against yolov8n-seg (/model.22) and yolo11n-seg
(/model.23): 10 outputs, correct shapes, finite, mask_i/proto bit-identical to
the raw cv4 conv / proto branch.
"""

from __future__ import annotations

import logging

import onnx

from .. import contract, onnx_helpers as oh
from ..identify import YoloIdentity
from .base import SurgeonBase
# shared v8 detection-base helpers live in pose.py (both task heads share the
# identical Detect base); reuse them here to stay DRY.
from .pose import _rewrite_detection, _expose, _head_conv_depth
from .yolo26 import SurgeonYolo26  # no-DFL/one2one detection base for yolo26-seg

log = logging.getLogger("model_surgery.surgeon.seg")


class SurgeonSeg(SurgeonBase):
    name = "yolov8-seg"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        head, H, W = ident.head_prefix, ident.height, ident.width
        # mask-coefficient count == prototype-mask channels (nm, default 32)
        nm = _head_conv_depth(model, f"{head}/cv4.0/cv4.0.2/Conv")
        # capture the proto-branch producer BEFORE _rewrite_detection clears outputs
        proto_node = self._proto_producer(model, ident)
        log.info("seg surgery on %s: %d mask coeffs, proto %dx%d",
                 head, nm, H // 4, W // 4)

        # 1) v8 detection base -> bbox_{i} + class_prob_{i}
        _rewrite_detection(model, ident)

        # 2) expose the 3 raw cv4 mask-coefficient heads as mask_{i} (no decode)
        for i, s in enumerate(contract.STRIDES):
            _expose(model, oh.find_node(model, f"{head}/cv4.{i}/cv4.{i}.2/Conv"), f"mask_{i}")
            oh.add_output(model, f"mask_{i}", (1, nm, H // s, W // s))

        # 3) expose the proto branch as proto
        _expose(model, proto_node, "proto")
        oh.add_output(model, "proto", (1, nm, H // 4, W // 4))

        return model

    @staticmethod
    def _proto_producer(model: onnx.ModelProto, ident: YoloIdentity) -> onnx.NodeProto:
        """The node emitting the (1, 32, H/4, W/4) prototype tensor. Prefer the
        graph output with that shape (robust to node renames); fall back to the
        canonical `{head}/proto/cv3/act/Mul`."""
        H, W = ident.height, ident.width
        for o in model.graph.output:
            d = [dim.dim_value for dim in o.type.tensor_type.shape.dim]
            if len(d) == 4 and d[2] == H // 4 and d[3] == W // 4:
                return oh.find_node_by_output(model, o.name)
        return oh.find_node(model, f"{ident.head_prefix}/proto/cv3/act/Mul")


class SurgeonSegV9(SurgeonSeg):
    name = "yolov9-seg"


class SurgeonSegV10(SurgeonSeg):
    name = "yolov10-seg"


class SurgeonSegV11(SurgeonSeg):
    name = "yolov11-seg"


class SurgeonYolo26Seg(SurgeonYolo26):
    """YOLO26 instance-segmentation surgery.

    Unlike v8/v9/v10/v11-seg (DFL detection base), yolo26-seg sits on the yolo26
    **no-DFL / one2one** Detect base. The mask branch is `{o}cv4.{i}` (32 raw
    coeffs, `o=one2one_`) plus the same `proto` branch. So surgery = the yolo26
    box+class rewrite (`SurgeonYolo26`, which also lowers attention + prunes the
    E2E tail) PLUS re-exposing `{o}cv4.{i}` as `mask_{i}` and the proto branch as
    `proto`. Adding those outputs BEFORE save keeps them reachable through the
    reachability prune. Contract: the same 10 tensors as v8-seg."""

    name = "yolo26-seg"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        head = ident.head_prefix
        o = ident.one2one_prefix or "one2one_"
        H, W = ident.height, ident.width
        nm = _head_conv_depth(model, f"{head}/{o}cv4.0/{o}cv4.0.2/Conv")
        proto_node = SurgeonSeg._proto_producer(model, ident)   # capture before outputs are cleared
        log.info("yolo26-seg surgery on %s: %d mask coeffs, proto %dx%d (no-DFL/one2one)",
                 head, nm, H // 4, W // 4)

        # 1) yolo26 no-DFL box + class base -> bbox_{i} + class_prob_{i} (clears outputs)
        super().do_surgery(model, ident)

        # 2) expose the 3 raw one2one_cv4 mask-coefficient heads as mask_{i}
        for i, s in enumerate(contract.STRIDES):
            _expose(model, oh.find_node(model, f"{head}/{o}cv4.{i}/{o}cv4.{i}.2/Conv"), f"mask_{i}")
            oh.add_output(model, f"mask_{i}", (1, nm, H // s, W // s))

        # 3) expose the proto branch as proto
        _expose(model, proto_node, "proto")
        oh.add_output(model, "proto", (1, nm, H // 4, W // 4))

        return model
