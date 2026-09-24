"""
YOLOv7 (anchor-based) detection-head surgery.

YOLOv7's `IDetect` head is anchor-based and, once exported to ONNX, is
structurally identical to YOLOv5's `Detect`: three per-scale output convs named
`<head>/m.{i}/Conv` of depth `na*(5+nc)` (255 for na=3, nc=80), each followed by
the same `Reshape -> Transpose -> Sigmoid -> Split -> (xy/wh decode) -> Concat
-> Reshape` tail that concatenates into a single `(1, 25200, 85)` output.

The SiMa box-decoder treats v5 and v7 with the same anchor-based path (only the
`decode_type` string differs: `"yolov7"`), applying sigmoid + the anchor/stride
formula + NMS itself. So the surgery is *exactly* the YOLOv5 surgery — re-expose
the three raw head convs as `raw_0`/`raw_1`/`raw_2` (stride 8/16/32) and let the
decode tail become unreachable and get pruned. We therefore subclass
`SurgeonYoloV5` and only change the registry `name` (mirroring how `yolo26`
reuses the `yolov8` box math).

Validated against a real `yolov7-tiny` graph (WongKinYiu/yolov7 topology) —
head at `/model/model.77`, three `m.{i}/Conv` of depth 255.
"""

from __future__ import annotations

import logging

from .yolov5 import SurgeonYoloV5

log = logging.getLogger("model_surgery.surgeon.yolov7")


class SurgeonYoloV7(SurgeonYoloV5):
    """YOLOv7 IDetect: identical anchor-based head, so identical surgery."""

    name = "yolov7"
