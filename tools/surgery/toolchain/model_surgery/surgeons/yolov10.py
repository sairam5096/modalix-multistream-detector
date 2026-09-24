"""
YOLOv10 detection-head surgery.

v10 exports its NMS-free *one2one* head (the one2many `cv2`/`cv3` training
branches are pruned at export) and, unlike YOLO26, that one2one head still uses
DFL. So surgery is v8's DFL unroll applied to the `one2one_` branches
(yolo26-style tapping):

  box : `{head}/one2one_cv2.{i}/one2one_cv2.{i}.2/Conv`  (64ch = 4x16 DFL bins)
        fed by `{head}/one2one_cv2.{i}/one2one_cv2.{i}.1/act/Mul`
  cls : `{head}/one2one_cv3.{i}/one2one_cv3.{i}.2/Conv`
  proj: the shared `{head}.dfl.conv.weight`

The box branch is unrolled into four per-side Conv+Softmax+Conv DFL branches,
concatenated to (l,t,r,b), run through the fixed LTRB->(cx,cy,w,h) conv (x
stride) and a grid/stride Add to land in input-pixel space; the class conv is
Sigmoided. The whole E2E TopK/Gather/GatherElements tail is left dangling from
the new outputs and dropped by `save_model`'s reachability prune. The PSA
attention block (`.../attn/qkv/conv/Conv`) sits upstream of the head and is
untouched.

Validated against yolov10n (Ultralytics 8.4.95, head `/model.23`, one2one_,
80 cls) — the surgered 6-tensor output (cx,cy,w,h converted to x1y1x2y2) matches
the model's built-in pre-NMS `(1,84,8400)` decode to box max|diff| ~1e-4,
class 0.
"""

from __future__ import annotations

import logging

import onnx

from .. import attention, contract, onnx_helpers as oh
from ..identify import YoloIdentity
from .base import SurgeonBase
from .yolov8 import _LTRB_TO_XYWH, _grid_offset  # identical DFL box math

log = logging.getLogger("model_surgery.surgeon.yolov10")


class SurgeonYoloV10(SurgeonBase):
    name = "yolov10"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        head = ident.head_prefix
        o = ident.one2one_prefix or "one2one_"
        dfl_w = ident.dfl_weight
        H, W, nc = ident.height, ident.width, ident.num_classes
        log.info("v10 surgery on head=%s one2one=%s dfl_weight=%s (%dx%d, %d cls)",
                 head, o, dfl_w, W, H, nc)

        # 0) lower any C2PSA MatMul attention to the MLA-supported Einsum form
        #    (upstream of the head; no-op if the model has none)
        attention.apply(model)

        if not oh.is_initializer(model, dfl_w):
            raise ValueError(f"DFL weight '{dfl_w}' not found — not a standard YOLOv10 DFL head")

        # 1) declare the contract outputs up-front
        oh.clear_outputs(model)
        for name, shape in contract.box_decoder_outputs(H, W, nc):
            oh.add_output(model, name, shape)

        # 2) bbox path — v8-style DFL unroll, tapped off the one2one_cv2 branch
        for i in range(len(contract.STRIDES)):
            stride = contract.STRIDES[i]
            cur_h, cur_w = H // stride, W // stride
            base = f"{head}/{o}cv2.{i}/{o}cv2.{i}.2"

            old_conv = oh.find_node(model, f"{base}/Conv")
            old_weight = oh.find_initializer_value(model, old_conv.input[1])   # (64,C,1,1)
            old_bias = oh.find_initializer_value(model, old_conv.input[2])     # (64,)
            mul_node = oh.find_node(model, f"{head}/{o}cv2.{i}/{o}cv2.{i}.1/act/Mul")

            dfl_convs: list[onnx.NodeProto] = [None] * 4  # type: ignore[list-item]
            anchor = mul_node
            for s in range(3, -1, -1):
                cw = f"{base}/{s}/Conv"
                oh.add_initializer(model, f"{cw}.weight", old_weight[16 * s:16 * (s + 1), ...])
                oh.add_initializer(model, f"{cw}.bias", old_bias[16 * s:16 * (s + 1)])
                conv = oh.make_node(
                    name=cw, op_type="Conv",
                    inputs=[mul_node.output[0], f"{cw}.weight", f"{cw}.bias"],
                    outputs=[f"{cw}_output"],
                )
                oh.insert_after(model, anchor, conv)

                sm_name = f"{head}/dfl/{i}/{s}/Softmax"
                softmax = oh.make_node(
                    name=sm_name, op_type="Softmax", axis=1,
                    inputs=conv.output, outputs=[f"{sm_name}_output"],
                )
                oh.insert_after(model, conv, softmax)

                proj_name = f"{head}/dfl/{i}/{s}/Conv"
                proj = oh.make_node(
                    name=proj_name, op_type="Conv",
                    inputs=[softmax.output[0], dfl_w], outputs=[f"{proj_name}_output"],
                )
                oh.insert_after(model, softmax, proj)
                dfl_convs[s] = proj
                anchor = proj

            concat_name = f"{head}/dfl/{i}/Concat"
            concat = oh.make_node(
                name=concat_name, op_type="Concat", axis=1,
                inputs=[c.output[0] for c in dfl_convs], outputs=[f"{concat_name}_output"],
            )
            oh.insert_after(model, dfl_convs[0], concat)

            conv_name = f"{head}/dfl/{i}/Conv"
            oh.add_initializer(model, f"{conv_name}.weight", _LTRB_TO_XYWH * stride)
            conv = oh.make_node(
                name=conv_name, op_type="Conv",
                inputs=[concat.output[0], f"{conv_name}.weight"], outputs=[f"{conv_name}_output"],
            )
            oh.insert_after(model, concat, conv)

            add_name = f"{head}/dfl/{i}/Add"
            oh.add_initializer(model, f"{add_name}.Const", _grid_offset(cur_h, cur_w, stride))
            add = oh.make_node(
                name=add_name, op_type="Add",
                inputs=[conv.output[0], f"{add_name}.Const"], outputs=[f"bbox_{i}"],
            )
            oh.insert_after(model, conv, add)

            oh.remove_node(model, f"{base}/Conv", tolerant=True)

        # 3) class path — Sigmoid straight off one2one_cv3.*.2/Conv
        for i in range(len(contract.STRIDES)):
            conv = oh.find_node(model, f"{head}/{o}cv3.{i}/{o}cv3.{i}.2/Conv")
            sig = oh.make_node(
                name=f"{head}/{o}cv3.{i}/{o}cv3.{i}.2/Sigmoid", op_type="Sigmoid",
                inputs=conv.output, outputs=[f"class_prob_{i}"],
            )
            oh.insert_after(model, conv, sig)

        # 4) the E2E TopK/Gather tail is now unreachable from the new outputs —
        #    save_model's keep_reachable_from_outputs() prunes it.
        return model
