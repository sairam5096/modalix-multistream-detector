"""
YOLOv8 detection-head surgery.

Rewrites the Ultralytics Detect head (`/model.<idx>/...`) so the graph exposes
the box-decoder contract: `bbox_{0,1,2}` (decoded, pixel-space cx,cy,w,h) and
`class_prob_{0,1,2}` (post-Sigmoid) at strides 8/16/32.

Adapted from the reference `surgeon_yolov8.py` (bbox_version==2 path) but driven
by the *auto-detected* head prefix (not hard-coded `/model.22`) and with
tolerant tail removal. The DFL is unrolled into four per-position Conv+Softmax+
Conv branches, then a fixed 4x4 conv converts (l,t,r,b) distances to (cx,cy,w,h)
and a grid-offset Add lands them in input-pixel space.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx

from .. import attention, contract, onnx_helpers as oh
from ..identify import YoloIdentity
from .base import SurgeonBase

log = logging.getLogger("model_surgery.surgeon.yolov8")

# (l,t,r,b) -> (cx_off, cy_off, w, h) : cx=(r-l)/2, cy=(b-t)/2, w=l+r, h=t+b
_LTRB_TO_XYWH = np.array(
    [[-0.5, 0, 0.5, 0],
     [0, -0.5, 0, 0.5],
     [1, 0, 1, 0],
     [0, 1, 0, 1]], dtype=np.float32,
).reshape(4, 4, 1, 1)


def _grid_offset(cur_h: int, cur_w: int, stride: int) -> np.ndarray:
    """(1,4,h,w): ch0=(0.5+col)*stride, ch1=(0.5+row)*stride, ch2=ch3=0."""
    xs = (np.arange(cur_w, dtype=np.float32) + 0.5) * stride
    ys = (np.arange(cur_h, dtype=np.float32) + 0.5) * stride
    ch0 = np.broadcast_to(xs[None, :], (cur_h, cur_w))
    ch1 = np.broadcast_to(ys[:, None], (cur_h, cur_w))
    zero = np.zeros((cur_h, cur_w), dtype=np.float32)
    return np.stack([ch0, ch1, zero, zero], axis=0)[None].astype(np.float32)


class SurgeonYoloV8(SurgeonBase):
    name = "yolov8"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        # Rewrite attention (MatMul -> Einsum) so it maps to the MLA instead of falling back
        # to the A65 (v12n was 17 MLA + 36 A65 stages without this). v10/v11/yolo26 have their
        # own surgeons that call attention.apply() before super(), so only run it here for the
        # BARE v8 surgeon — i.e. v8 (no attention -> no-op) and v12 (which has no dedicated
        # surgeon and routes through this base). The type guard avoids a double-apply, which
        # crashes when the second pass looks for attention nodes the first pass already removed.
        if type(self) is SurgeonYoloV8:
            attention.apply(model)
        head = ident.head_prefix
        dfl_w = ident.dfl_weight
        H, W, nc = ident.height, ident.width, ident.num_classes
        log.info("v8 surgery on head=%s dfl_weight=%s (%dx%d, %d cls)", head, dfl_w, W, H, nc)

        if not oh.is_initializer(model, dfl_w):
            raise ValueError(f"DFL weight '{dfl_w}' not found — not a standard YOLOv8 DFL head")

        # 1) declare the contract outputs up-front
        oh.clear_outputs(model)
        for name, shape in contract.box_decoder_outputs(H, W, nc):
            oh.add_output(model, name, shape)

        # 2) bbox path — unroll DFL per scale
        for i in range(len(contract.STRIDES)):
            stride = contract.STRIDES[i]
            cur_h, cur_w = H // stride, W // stride
            base = f"{head}/cv2.{i}/cv2.{i}.2"

            old_conv = oh.find_node(model, f"{base}/Conv")
            old_weight = oh.find_initializer_value(model, old_conv.input[1])   # (64,C,1,1)
            old_bias = oh.find_initializer_value(model, old_conv.input[2])     # (64,)
            mul_node = oh.find_node(model, f"{head}/cv2.{i}/cv2.{i}.1/act/Mul")

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

        # 3) class path — Sigmoid straight off cv3.*.2/Conv
        for i in range(len(contract.STRIDES)):
            conv = oh.find_node(model, f"{head}/cv3.{i}/cv3.{i}.2/Conv")
            sig = oh.make_node(
                name=f"{head}/cv3.{i}/cv3.{i}.2/Sigmoid", op_type="Sigmoid",
                inputs=conv.output, outputs=[f"class_prob_{i}"],
            )
            oh.insert_after(model, conv, sig)

        # 4) delete the original decode tail (tolerant — export naming varies slightly)
        tail = [
            "Slice_1", "Sigmoid", "Concat", "Concat_1", "Concat_2", "Concat_3",
            "Concat_4", "Concat_5", "Reshape", "Reshape_1", "Reshape_2", "Slice",
            "dfl/Reshape", "dfl/Transpose", "dfl/Softmax", "dfl/conv/Conv",
            "dfl/Reshape_1", "Split", "Add_1", "Add_2", "Sub", "Sub_1",
            "Div_1", "Mul_2",
        ]
        removed = oh.remove_nodes(model, [f"{head}/{n}" for n in tail])
        log.info("removed %d/%d original tail nodes", len(removed), len(tail))
        return model
