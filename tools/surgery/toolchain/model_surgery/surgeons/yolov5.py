"""
YOLOv5 (anchor-based) detection-head surgery.

Unlike the anchor-free families (v8/v10/v11/yolo26) — which we decode *in the
graph* into the 6-tensor `bbox_/class_prob_` contract — the SiMa generic
box-decoder handles YOLOv5/v7 with its **anchor-based** path: it takes the three
*raw* detection-head conv outputs and applies sigmoid + the anchor/stride
formula + NMS itself (see `genericboxdecode` `anchorbased_v5_v7.json`,
`decode_type="yolov5"`/`"yolov7"`).

So surgery here does **not** decode anything. A stock YOLOv5 export ends its
Detect head (`/model.24`) with, per scale:

    /model.24/m.{i}/Conv   (1, na*(5+nc), H/s, W/s)   # <-- the raw tensor we want
        -> Reshape -> Transpose -> Sigmoid -> Split -> (xy/wh decode) -> Concat
        -> Reshape -> ... -> Concat_3 -> output0  (1, 25200, 85)

All this surgeon has to do is **re-expose those three raw head convs** as the
graph outputs `raw_0`/`raw_1`/`raw_2` (stride 8/16/32 order) and detach the
model's own sigmoid/anchor-grid/decode/concat tail. Because we re-point the
graph outputs at the conv tensors and drop `output0`, the whole decode tail
becomes unreachable and `save_model()`'s reachability prune deletes it — no
explicit tail enumeration needed.

Contract produced (3 tensors, raw logits — `class_is_prob=false`):

    raw_0 : (1, na*(5+nc), H/8,  W/8)     # e.g. (1, 255, 80, 80) for na=3, nc=80
    raw_1 : (1, na*(5+nc), H/16, W/16)    #      (1, 255, 40, 40)
    raw_2 : (1, na*(5+nc), H/32, W/32)    #      (1, 255, 20, 20)

Verified against the stock Ultralytics `yolov5s.onnx` (legacy anchor-based
export) — head at `/model.24`, three `m.{i}/Conv` of depth 255.
"""

from __future__ import annotations

import logging

import onnx
from onnx import helper as onnx_helper

from .. import contract, onnx_helpers as oh
from ..identify import YoloIdentity
from .base import SurgeonBase

log = logging.getLogger("model_surgery.surgeon.yolov5")

# Canonical anchor-based contract output names (stride 8/16/32 order).
RAW_PREFIX = "raw_"


class SurgeonYoloV5(SurgeonBase):
    """Re-expose the three raw anchor-based head convs; strip the decode tail."""

    name = "yolov5"

    # Head-conv name template. YOLOv5's Detect and YOLOv7's IDetect both name the
    # per-scale output conv `<head>/m.<i>/Conv`, so v7 reuses this unchanged.
    conv_template = "{head}/m.{i}/Conv"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        head = ident.head_prefix
        H, W, nc = ident.height, ident.width, ident.num_classes
        strides = contract.STRIDES  # (8, 16, 32) — matches m.0/m.1/m.2 (P3/P4/P5)
        log.info("%s surgery on head=%s (%dx%d, %d cls) — expose %d raw head convs",
                 self.name, head, W, H, nc, len(strides))

        # 1) locate the three raw head convs; read their true output depth + dtype.
        head_convs: list[onnx.NodeProto] = []
        depths: list[int] = []
        elem_types: list[int] = []
        for i in range(len(strides)):
            conv_name = self.conv_template.format(head=head, i=i)
            conv = oh.find_node(model, conv_name)          # KeyError if missing
            weight = oh.find_initializer_value(model, conv.input[1])  # (depth,C,1,1)
            depth = int(weight.shape[0])
            # Sanity: an anchor-based head packs na copies of (5 + nc) per cell.
            if depth % (5 + nc) != 0:
                raise ValueError(
                    f"head conv '{conv_name}' depth {depth} is not a multiple of "
                    f"(5 + num_classes)={5 + nc} — not a stock anchor-based "
                    f"YOLOv5/v7 head (expected na*(5+nc), e.g. 255 for na=3, nc=80)")
            na = depth // (5 + nc)
            log.info("  scale %d: %s depth=%d (na=%d, stride=%d)",
                     i, conv_name, depth, na, strides[i])
            head_convs.append(conv)
            depths.append(depth)
            # Preserve the conv's own dtype (fp32 or fp16) — a Conv's output type
            # equals its weight type. oh.add_output declares FLOAT, so we patch the
            # value-info below for fp16 exports (otherwise ORT flags a type error).
            elem_types.append(onnx_helper.np_dtype_to_tensor_dtype(weight.dtype))

        # 2) re-point the graph outputs at those conv tensors as raw_0/1/2.
        oh.clear_outputs(model)
        for i, (conv, depth) in enumerate(zip(head_convs, depths)):
            stride = strides[i]
            cur_h, cur_w = H // stride, W // stride
            new_name = f"{RAW_PREFIX}{i}"
            old_name = conv.output[0]

            # Rename the conv's output tensor to the canonical contract name and
            # rewire every consumer of the old name (the decode tail) so the graph
            # stays well-formed; the tail is unreachable from raw_* and is pruned.
            conv.output[0] = new_name
            for node in model.graph.node:
                for j, inp in enumerate(node.input):
                    if inp == old_name:
                        node.input[j] = new_name

            oh.add_output(model, new_name, (1, depth, cur_h, cur_w))
            model.graph.output[-1].type.tensor_type.elem_type = elem_types[i]

        # 3) `output0` and the sigmoid/anchor-grid/decode/concat tail no longer
        #    feed any graph output — save_model()'s keep_reachable_from_outputs()
        #    prunes the entire tail.
        return model
