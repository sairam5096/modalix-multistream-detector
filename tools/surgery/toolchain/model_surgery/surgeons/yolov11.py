"""
YOLOv11 detection-head surgery.

YOLO11's Detect head is structurally identical to v8's DFL head — box branch
`{head}/cv2.{i}/cv2.{i}.2/Conv` fed by `{head}/cv2.{i}/cv2.{i}.1/act/Mul`, class
branch `{head}/cv3.{i}/cv3.{i}.2/Conv`, projection `{head}.dfl.conv.weight` — at
its own `/model.N` index (typically `/model.23`). The v11-specific parts (C3k2
backbone, the C2PSA attention block `.../m/m.0/attn/qkv/conv/Conv`) all sit
*upstream* of the head and are untouched by surgery; they run as-is under
onnxruntime. So the v8 surgery, driven by the auto-detected `ident.head_prefix`
+ `ident.dfl_weight`, applies verbatim. We keep a distinct registered surgeon so
dispatch/metadata report the correct family.

Validated against yolo11n (Ultralytics 8.4.95, head `/model.23`, 80 cls) — the
surgered 6-tensor output matches the model's built-in `(1,84,8400)` decode to
box max|diff| ~1e-4, class 0.
"""

from __future__ import annotations

import onnx

from .. import attention
from ..identify import YoloIdentity
from .yolov8 import SurgeonYoloV8


class SurgeonYoloV11(SurgeonYoloV8):
    name = "yolov11"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        # The C2PSA attention block (`.../attn/qkv/conv/Conv`) sits upstream of
        # the head and lowers to MatMul, which hangs the MLA. Rewrite it to the
        # supported Einsum form first (no-op if absent), then run the v8 DFL head
        # surgery verbatim.
        attention.apply(model)
        return super().do_surgery(model, ident)
