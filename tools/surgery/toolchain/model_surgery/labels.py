"""
Class-label handling.

Labels must travel *inside* the model (embedded in ONNX metadata) so the Nx AI
Manager plugin can read them straight from the model file — no external sidecar
required at inference time. We still emit them in `boxdecoder.json` for the
pipeline builder, but the model is the source of truth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("model_surgery.labels")

# COCO 80 (Ultralytics naming — matches the runtime's expected label set).
COCO80 = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "potted plant", "bed", "dining table", "toilet", "TV monitor", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# The output box format the SiMa box-decoder / runtime emits per detection:
# x1, y1, x2, y2, score, classid.
BBOX_FORMAT = "xyxysc"


def load_labels(source: str | Path | None, num_classes: int) -> list[str]:
    """
    Resolve the label list.

    `source` may be a path to a .txt (one label per line), a .json (list or
    {"labels": [...]}), or None. When None (or on a count mismatch) we fall back
    to COCO80 if it fits, otherwise synthesize class_0..class_{n-1}.
    """
    labels: list[str] | None = None

    if source is not None:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"labels file not found: {p}")
        text = p.read_text(encoding="utf-8").strip()
        if p.suffix.lower() == ".json":
            data = json.loads(text)
            labels = data["labels"] if isinstance(data, dict) else list(data)
        else:
            labels = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if labels is None:
        if num_classes == len(COCO80):
            log.info("no labels supplied and num_classes==80 -> using COCO80")
            labels = list(COCO80)
        else:
            log.warning(
                "no labels supplied for %d classes -> synthesizing class_<i>", num_classes
            )
            labels = [f"class_{i}" for i in range(num_classes)]

    if len(labels) != num_classes:
        log.warning(
            "label count (%d) != model num_classes (%d); truncating/padding to match",
            len(labels), num_classes,
        )
        if len(labels) > num_classes:
            labels = labels[:num_classes]
        else:
            labels += [f"class_{i}" for i in range(len(labels), num_classes)]

    return labels


def formatted_label_string(labels: list[str]) -> str:
    """
    Build the runtime's single-string label encoding:
        "bboxes-format:xyxysc;0:person;1:bicycle;...;79:toothbrush"
    This is what the SiMa Neat runtime places in tensors->names[0].
    """
    parts = [f"bboxes-format:{BBOX_FORMAT}"]
    parts += [f"{i}:{name}" for i, name in enumerate(labels)]
    return ";".join(parts)
