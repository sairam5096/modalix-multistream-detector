"""
YOLO26 detection-head surgery.

YOLO26 dropped DFL and ships an E2E (one2one) head that exports to a `(1,300,6)`
NMS-free output. For the SiMa box-decoder we don't want that E2E tail — we want
the same 6-tensor `cxcywh_pixel` contract as v8. Because there is no DFL, the
`one2one_cv2.{i}.2/Conv` already emits the 4 `[l,t,r,b]` distances directly, so
surgery is *v8 minus the DFL unroll*: feed those 4 channels straight into the
LTRB→(cx,cy,w,h) decode + grid/stride offset, Sigmoid the class conv, and let the
reachability prune drop the entire TopK/Gather E2E tail.

Verified against yolo26n (Ultralytics 8.4.x) — head at `/model.23`, one2one_cv2/
one2one_cv3 branches, box depth 4.
"""

from __future__ import annotations

import logging

import onnx

from .. import attention, contract, onnx_helpers as oh
from ..identify import YoloIdentity
from .base import SurgeonBase
from .yolov8 import _LTRB_TO_XYWH, _grid_offset  # identical box math, no DFL

log = logging.getLogger("model_surgery.surgeon.yolo26")


class SurgeonYolo26(SurgeonBase):
    name = "yolo26"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        head = ident.head_prefix
        o = ident.one2one_prefix or "one2one_"
        H, W, nc = ident.height, ident.width, ident.num_classes
        log.info("yolo26 surgery on head=%s (one2one=%s, %dx%d, %d cls, no-DFL)",
                 head, o, W, H, nc)

        # 0) lower any C2PSA MatMul attention to the MLA-supported Einsum form
        #    (upstream of the head; no-op if the model has none)
        attention.apply(model)

        # sanity: box branch must be depth-4 (no DFL)
        box_w = f"{head[1:]}.{o}cv2.2.2.weight"
        if oh.is_initializer(model, box_w):
            depth = int(oh.find_initializer_value(model, box_w).shape[0])
            if depth != 4:
                raise ValueError(f"expected depth-4 box head for yolo26, got {depth} "
                                 f"(is this really a DFL-free model?)")

        # 1) declare the contract outputs
        oh.clear_outputs(model)
        for name, shape in contract.box_decoder_outputs(H, W, nc):
            oh.add_output(model, name, shape)

        # 2) bbox path — no DFL: decode straight off the cv2 conv (l,t,r,b)
        for i in range(len(contract.STRIDES)):
            stride = contract.STRIDES[i]
            cur_h, cur_w = H // stride, W // stride

            box_conv = oh.find_node(model, f"{head}/{o}cv2.{i}/{o}cv2.{i}.2/Conv")

            conv_name = f"{head}/decode/{i}/Conv"
            oh.add_initializer(model, f"{conv_name}.weight", _LTRB_TO_XYWH * stride)
            conv = oh.make_node(
                name=conv_name, op_type="Conv",
                inputs=[box_conv.output[0], f"{conv_name}.weight"],
                outputs=[f"{conv_name}_output"],
            )
            oh.insert_after(model, box_conv, conv)

            add_name = f"{head}/decode/{i}/Add"
            oh.add_initializer(model, f"{add_name}.Const", _grid_offset(cur_h, cur_w, stride))
            add = oh.make_node(
                name=add_name, op_type="Add",
                inputs=[conv.output[0], f"{add_name}.Const"], outputs=[f"bbox_{i}"],
            )
            oh.insert_after(model, conv, add)

        # 3) class path — Sigmoid off cv3.*.2/Conv
        for i in range(len(contract.STRIDES)):
            cls_conv = oh.find_node(model, f"{head}/{o}cv3.{i}/{o}cv3.{i}.2/Conv")
            sig = oh.make_node(
                name=f"{head}/{o}cv3.{i}/{o}cv3.{i}.2/Sigmoid", op_type="Sigmoid",
                inputs=cls_conv.output, outputs=[f"class_prob_{i}"],
            )
            oh.insert_after(model, cls_conv, sig)

        # 4) the E2E TopK/Gather tail is now unreachable from the outputs —
        #    save_model's keep_reachable_from_outputs() prunes it.
        return model
