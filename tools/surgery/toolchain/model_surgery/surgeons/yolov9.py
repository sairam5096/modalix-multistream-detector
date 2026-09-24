"""
YOLOv9 detection-head surgery.

A standard Ultralytics YOLOv9 export (yolov9t/s/c/e) carries a *v8-style DFL
Detect head*: the box branch is `{head}/cv2.{i}/cv2.{i}.2/Conv` (64ch = 4x16 DFL
bins) fed by `{head}/cv2.{i}/cv2.{i}.1/act/Mul`, the class branch is
`{head}/cv3.{i}/cv3.{i}.2/Conv`, and the projection is `{head}.dfl.conv.weight`
— identical naming to v8, only at a different `/model.N` index (typically
`/model.22`). Since `SurgeonYoloV8.do_surgery` is driven entirely by the
auto-detected `ident.head_prefix` + `ident.dfl_weight`, the v8 surgery applies
verbatim; we keep a distinct registered surgeon so dispatch/metadata report the
correct family.

Validated against yolov9t (Ultralytics 8.4.95, head `/model.22`, 80 cls) — the
surgered 6-tensor output matches the model's built-in `(1,84,8400)` decode to
box max|diff| ~1e-4, class 0.
"""

from __future__ import annotations

from .yolov8 import SurgeonYoloV8


class SurgeonYoloV9(SurgeonYoloV8):
    name = "yolov9"
