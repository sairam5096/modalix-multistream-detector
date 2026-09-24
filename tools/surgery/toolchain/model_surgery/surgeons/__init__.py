"""Surgeon plugins. Importing this package registers all built-in surgeons."""

from .base import SurgeonBase, get_surgeon, registry  # noqa: F401
# import each module so its SurgeonBase subclass self-registers
from . import (  # noqa: F401
    yolov5, yolov6, yolov7, yolov8, yolov9, yolov10, yolov11, yolo26, yolox,
    pose, seg, obb, classify, efficientdet, centernet,
)
