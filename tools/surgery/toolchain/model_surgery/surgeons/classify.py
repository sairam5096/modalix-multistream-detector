"""
YOLOv8 / v11 (/v12 / yolo26) image-classification surgery.

An Ultralytics `-cls` model has NO detection head: the backbone feeds a
`GlobalAveragePool -> Flatten -> Gemm (-> Softmax)` classifier that emits a single
rank-2 `(1, nc)` score vector (post-Softmax probabilities in the stock export).
There is no DFL, no anchors, no box/pose/mask branch, so surgery is a near
pass-through — the graph already produces exactly the tensor the SiMa classifier
post-processor wants. All we do is:

  1. lower any upstream C2PSA attention (`.../attn/qkv/conv/Conv`, present on
     v11/v12-cls backbones) from MatMul to the MLA-supported Einsum form — the
     same fix the detect surgeons apply, so the MLA compiler doesn't hang;
  2. re-expose the final score tensor under the canonical name `class_scores`
     (declaring it `(1, nc)`), and let the reachability prune drop nothing else.

Contract — 1 tensor:
    class_scores    (1, nc)    per-class scores (Softmax probabilities)

Top-1 = argmax(class_scores); labels travel in the model metadata. Validated
(Ultralytics 8.4.104) against yolov8n-cls (/model.9, no attention) and
yolo11n-cls (/model.10, C2PSA attention) — 1 output, shape (1, 1000), finite,
argmax bit-matches the stock `.onnx` classifier output.
"""

from __future__ import annotations

import logging

import onnx

from .. import attention, onnx_helpers as oh
from ..identify import YoloIdentity
from .base import SurgeonBase
from .pose import _expose            # canonical re-name of an existing producer

log = logging.getLogger("model_surgery.surgeon.classify")


class SurgeonClassify(SurgeonBase):
    name = "yolov8-cls"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        nc = ident.num_classes
        log.info("classify surgery: %d classes -> single 'class_scores' vector", nc)

        # 1) lower C2PSA MatMul attention to Einsum (no-op on v8-cls; needed for
        #    v11/v12-cls whose backbone carries the C2PSA block).
        attention.apply(model)

        # 2) re-expose the single (1, nc) classifier output as `class_scores`.
        if len(model.graph.output) != 1:
            raise ValueError(f"classify model expected 1 output, got {len(model.graph.output)}")
        producer = oh.find_node_by_output(model, model.graph.output[0].name)
        oh.clear_outputs(model)
        _expose(model, producer, "class_scores")
        oh.add_output(model, "class_scores", (1, nc))
        return model


class SurgeonClassifyV11(SurgeonClassify):
    """v11-cls — identical classifier tail at its own /model.N; kept distinct so
    dispatch/metadata report `yolov11-cls`. (v12/yolo26-cls route through one of
    these two via the backbone-attention signature — surgery is identical.)"""
    name = "yolov11-cls"


class SurgeonDepth(SurgeonBase):
    """Depth-estimation surgery: op-compat ONLY. Lower C2PSA MatMul attention to Einsum (single-MLA)
    and keep the single DENSE output (1,1,h,w) as `depth` — NO box decoder (the SiMa compiler emits
    the raw dense map). Accuracy is scored off-graph with a depth metric (RMSE/delta), not mAP."""

    name = "depth"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        log.info("depth surgery: attention lowering + keep dense output (no box decoder)")
        attention.apply(model)
        if len(model.graph.output) != 1:
            raise ValueError(f"depth model expected 1 output, got {len(model.graph.output)}")
        out = model.graph.output[0]
        dims = tuple(d.dim_value for d in out.type.tensor_type.shape.dim)
        producer = oh.find_node_by_output(model, out.name)
        oh.clear_outputs(model)
        _expose(model, producer, "depth")
        oh.add_output(model, "depth", dims)
        return model
