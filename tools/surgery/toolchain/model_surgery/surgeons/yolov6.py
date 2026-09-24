"""
YOLOv6 (Meituan) detection-head surgery.

Rewrites the Meituan YOLOv6 3.0+ *efficient decoupled head* (`detect.*` module)
into the SiMa box-decoder contract: `bbox_{0,1,2}` (decoded, pixel-space
cx,cy,w,h) + `class_prob_{0,1,2}` (post-Sigmoid) at strides 8/16/32.

Unlike Ultralytics exports, Meituan node names are auto-numbered (`Conv_160`,
`Conv_289`, ...) and carry *no* semantic path — so this surgeon locates the head
convolutions by their **weight-initializer name** (`detect.cls_preds.{i}.weight`,
`detect.reg_preds*.{i}.weight`) rather than by node name.

YOLOv6 ships two head flavours and this surgeon handles both automatically:

  * **no-DFL** (n / s): `detect.reg_preds_lrtb.{i}` emits the 4 `[l,t,r,b]`
    distances directly — tap it straight into the LTRB->cxcywh decode
    (yolo26-style, no unroll).
  * **DFL** (m / l): `detect.reg_preds.{i}` emits `4*reg_max` bins projected by
    `detect.proj_conv.weight` (reg_max is read from that weight — YOLOv6 uses
    **17**, cf. v8's 16). The DFL is unrolled per side into
    Conv(bin-slice)+Softmax+Conv(proj) exactly like the v8 surgeon, then
    concatenated to 4 `[l,t,r,b]` channels.

Both flavours then share the v8 box math: a fixed 4x4 conv maps `(l,t,r,b)` ->
`(cx,cy,w,h)*stride` and a grid-offset Add lands them in input-pixel space —
numerically identical to YOLOv6's own `dist2bbox(anchor_points +/- ltrb)*stride`.
The original decode tail (Reshape/Transpose/Split/Sub/Add/Div dist2bbox block)
is left detached from the graph outputs and pruned by `save_model`'s
reachability sweep — no name-based tail deletion needed (names aren't stable).

Validated numerically against Meituan yolov6n (no-DFL) and yolov6m (DFL),
release 0.3.0, 640x640 / 80-class exports. See module-level surgery notes.
"""

from __future__ import annotations

import logging

import onnx

from .. import contract, onnx_helpers as oh
from ..identify import YoloIdentity
from .base import SurgeonBase
from .yolov8 import _LTRB_TO_XYWH, _grid_offset  # identical box math (LTRB->cxcywh + grid)

log = logging.getLogger("model_surgery.surgeon.yolov6")


def _find_conv_by_weight(model: onnx.ModelProto, weight_name: str) -> onnx.NodeProto:
    """Meituan node names are auto-numbered — find the Conv by its weight input."""
    for n in model.graph.node:
        if n.op_type == "Conv" and len(n.input) > 1 and n.input[1] == weight_name:
            return n
    raise KeyError(f"no Conv consumes weight initializer '{weight_name}' "
                   "(is this a Meituan YOLOv6 decoupled head?)")


def _reg_weight_name(model: onnx.ModelProto, hp: str, i: int, has_dfl: bool) -> str:
    """Resolve the per-scale regression conv weight (naming varies by flavour)."""
    if has_dfl:
        return f"{hp}.reg_preds.{i}.weight"
    for cand in (f"{hp}.reg_preds_lrtb.{i}.weight", f"{hp}.reg_preds.{i}.weight"):
        if oh.is_initializer(model, cand):
            return cand
    raise KeyError(f"no no-DFL regression weight for scale {i} under '{hp}'")


class SurgeonYoloV6(SurgeonBase):
    name = "yolov6"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        hp = ident.head_prefix.strip("/") or "detect"  # initializer prefix, e.g. "detect"
        H, W, nc = ident.height, ident.width, ident.num_classes

        # DFL is present iff the projection conv exists; reg_max comes from its shape
        # (YOLOv6 = 17, not v8's 16). This is the DFL-vs-no-DFL discriminator.
        proj_w = f"{hp}.proj_conv.weight"
        has_dfl = oh.is_initializer(model, proj_w)
        reg_max = int(oh.find_initializer_value(model, proj_w).shape[1]) if has_dfl else 0
        log.info("v6 surgery head=%s %dx%d nc=%d %s%s", hp, W, H, nc,
                 "DFL" if has_dfl else "no-DFL",
                 f" reg_max={reg_max}" if has_dfl else "")

        # 1) declare the contract outputs up-front
        oh.clear_outputs(model)
        for name, shape in contract.box_decoder_outputs(H, W, nc):
            oh.add_output(model, name, shape)

        for i in range(len(contract.STRIDES)):
            stride = contract.STRIDES[i]
            cur_h, cur_w = H // stride, W // stride

            # ---- bbox path: produce a (1,4,h,w) [l,t,r,b] tensor, then decode ----
            reg_conv = _find_conv_by_weight(model, _reg_weight_name(model, hp, i, has_dfl))

            if has_dfl:
                # unroll DFL per side: Conv(17-bin slice)+Softmax+Conv(proj) -> 1 ch
                feat = reg_conv.input[0]                                    # reg-branch feature
                reg_w = oh.find_initializer_value(model, reg_conv.input[1])  # (4*reg_max,C,1,1)
                reg_b = oh.find_initializer_value(model, reg_conv.input[2])  # (4*reg_max,)
                sides: list[onnx.NodeProto] = []
                anchor: onnx.NodeProto = reg_conv
                for s in range(4):
                    cw = f"{hp}/dfl/{i}/{s}/Conv"
                    oh.add_initializer(model, f"{cw}.weight", reg_w[reg_max * s:reg_max * (s + 1)])
                    oh.add_initializer(model, f"{cw}.bias", reg_b[reg_max * s:reg_max * (s + 1)])
                    conv = oh.make_node(
                        name=cw, op_type="Conv",
                        inputs=[feat, f"{cw}.weight", f"{cw}.bias"], outputs=[f"{cw}_output"],
                    )
                    oh.insert_after(model, anchor, conv)

                    sm_name = f"{hp}/dfl/{i}/{s}/Softmax"
                    softmax = oh.make_node(
                        name=sm_name, op_type="Softmax", axis=1,
                        inputs=conv.output, outputs=[f"{sm_name}_output"],
                    )
                    oh.insert_after(model, conv, softmax)

                    proj_name = f"{hp}/dfl/{i}/{s}/Proj"
                    proj = oh.make_node(
                        name=proj_name, op_type="Conv",
                        inputs=[softmax.output[0], proj_w], outputs=[f"{proj_name}_output"],
                    )
                    oh.insert_after(model, softmax, proj)
                    sides.append(proj)
                    anchor = proj

                concat_name = f"{hp}/dfl/{i}/Concat"
                ltrb_src: onnx.NodeProto = oh.make_node(
                    name=concat_name, op_type="Concat", axis=1,
                    inputs=[p.output[0] for p in sides], outputs=[f"{concat_name}_output"],
                )
                oh.insert_after(model, sides[-1], ltrb_src)
            else:
                # no-DFL: reg conv already emits the 4 [l,t,r,b] channels directly
                ltrb_src = reg_conv

            conv_name = f"{hp}/decode/{i}/Conv"
            oh.add_initializer(model, f"{conv_name}.weight", _LTRB_TO_XYWH * stride)
            dec = oh.make_node(
                name=conv_name, op_type="Conv",
                inputs=[ltrb_src.output[0], f"{conv_name}.weight"], outputs=[f"{conv_name}_output"],
            )
            oh.insert_after(model, ltrb_src, dec)

            add_name = f"{hp}/decode/{i}/Add"
            oh.add_initializer(model, f"{add_name}.Const", _grid_offset(cur_h, cur_w, stride))
            add = oh.make_node(
                name=add_name, op_type="Add",
                inputs=[dec.output[0], f"{add_name}.Const"], outputs=[f"bbox_{i}"],
            )
            oh.insert_after(model, dec, add)

            # ---- class path: Sigmoid straight off cls_preds.{i}/Conv ----
            cls_conv = _find_conv_by_weight(model, f"{hp}.cls_preds.{i}.weight")
            sig = oh.make_node(
                name=f"{hp}/cls/{i}/Sigmoid", op_type="Sigmoid",
                inputs=cls_conv.output, outputs=[f"class_prob_{i}"],
            )
            oh.insert_after(model, cls_conv, sig)

        # The original dist2bbox tail no longer feeds any graph output — it is
        # unreachable and save_model's keep_reachable_from_outputs() prunes it.
        return model
