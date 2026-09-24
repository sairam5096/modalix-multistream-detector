"""
YOLOX detection-head surgery.

YOLOX is anchor-free with a *decoupled* head: at each FPN scale (strides 8/16/32)
three parallel conv branches emit

    reg  : (1, 4,  h, w)   raw [tx, ty, tw, th]
    obj  : (1, 1,  h, w)   objectness   (already Sigmoid-ed in the export)
    cls  : (1, nc, h, w)   class scores (already Sigmoid-ed in the export)

which the standard export concatenates channel-wise into a per-scale
`(1, 4+1+nc, h, w)` tensor (order **reg, obj, cls**), then flattens each scale to
`(1, 85, h*w)`, stacks the scales to `(1, 85, 8400)` and transposes to the single
`(1, 8400, 85)` decode output.

The SiMa box-decoder (`decode_type="yolox"`, MODELS.md YOLOX row) wants the
**3-tensor** contract instead: the three per-scale `(1, 85, h, w)` maps, raw. It
applies the `(tx+col)*stride` / `exp(tw)*stride` decode, NMS (and Sigmoid when
`class_is_prob=false`) itself — so this surgeon does **not** decode. It simply
re-exposes each scale's [reg, obj, cls] branch outputs as a fresh `(1, 85, h, w)`
Concat named `raw_0`/`raw_1`/`raw_2` (strides 8/16/32). The original
flatten / scale-concat / transpose decode tail then feeds no output and is dropped
by `save_model`'s reachability prune.

No DFL, no anchor grids, no Ultralytics `cv2.*/cv3.*` head — the head is located
purely structurally: the three rank-4, axis-1 `Concat` nodes with 3 inputs of
channel depth `[4, 1, nc]`, ordered by spatial size (largest == stride 8).

Verified against yolox_s (640x640, 80 classes). Because obj/cls are already
sigmoided in the export, the exposed tensors are class-probabilities
(`class_is_prob=true`); a raw-logit export would simply expose logits instead
(`class_is_prob=false`) with no change to this surgery.
"""

from __future__ import annotations

import logging

import onnx
from onnx import shape_inference

from .. import onnx_helpers as oh
from ..identify import YoloIdentity
from .base import SurgeonBase

log = logging.getLogger("model_surgery.surgeon.yolox")

# stride per exposed tensor: raw_0 -> 8, raw_1 -> 16, raw_2 -> 32
STRIDES = (8, 16, 32)
RAW_PREFIX = "raw_"


def _shapes(model: onnx.ModelProto) -> dict[str, list[int | None]]:
    """name -> dim list (None for dynamic) for all typed tensors, shapes inferred."""
    vi: dict[str, list[int | None]] = {}
    for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        vi[v.name] = [
            d.dim_value if d.HasField("dim_value") else None
            for d in v.type.tensor_type.shape.dim
        ]
    return vi


class SurgeonYoloX(SurgeonBase):
    name = "yolox"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        H, W, nc = ident.height, ident.width, ident.num_classes
        depth = 4 + 1 + nc  # reg(4) + obj(1) + cls(nc) == 85 for COCO
        log.info("yolox surgery (%dx%d, %d cls, depth=%d, decoupled anchor-free head)",
                 W, H, nc, depth)

        # Raw YOLOX exports carry no intermediate value_info; infer shapes on a copy
        # so we can locate the head branches by channel depth (names differ per export).
        try:
            inferred = shape_inference.infer_shapes(model)
        except Exception as exc:  # pragma: no cover - malformed model
            raise ValueError(f"shape inference failed; cannot locate YOLOX head: {exc}")
        vi = _shapes(inferred)

        def in_depth(t: str) -> int | None:
            s = vi.get(t)
            return s[1] if s and len(s) >= 2 else None

        # The decoupled head is the only place with rank-4, axis-1 Concat nodes that
        # have exactly 3 inputs of channel depth [reg=4, obj=1, cls=nc]. Everything
        # else (focus, SPP, neck feature-fusion) has 2 or 4 inputs / other depths.
        heads: list[tuple[int, int, int, str, list[str]]] = []  # (area,h,w,concat_name,[reg,obj,cls])
        for n in inferred.graph.node:
            if n.op_type != "Concat" or len(n.input) != 3:
                continue
            axis = next((a.i for a in n.attribute if a.name == "axis"), None)
            osh = vi.get(n.output[0])
            if axis != 1 or not osh or len(osh) != 4 or osh[1] != depth:
                continue
            if [in_depth(t) for t in n.input] != [4, 1, nc]:
                continue
            h, w = osh[2], osh[3]
            heads.append((h * w, h, w, n.name, list(n.input)))

        if len(heads) != 3:
            raise ValueError(
                f"expected 3 YOLOX decoupled-head Concats (rank-4, axis=1, inputs "
                f"[4,1,{nc}]), found {len(heads)} — not a standard raw YOLOX export "
                f"(already surgeoned or wrong family?)")

        heads.sort(key=lambda x: -x[0])  # largest spatial map first -> stride 8

        # Re-expose: one fresh Concat[reg, obj, cls] -> raw_i per scale. The original
        # per-scale concat + flatten/scale-concat/transpose tail becomes unreachable.
        oh.clear_outputs(model)
        summary = []
        for i, (_, h, w, concat_name, (reg, obj, cls)) in enumerate(heads):
            stride = STRIDES[i]
            exp_h, exp_w = H // stride, W // stride
            if (h, w) != (exp_h, exp_w):
                log.warning("scale %d map is %dx%d, expected %dx%d for stride %d",
                            i, h, w, exp_h, exp_w, stride)

            name = f"{RAW_PREFIX}{i}"
            new = oh.make_node(
                name=f"/yolox/head/{i}/Concat", op_type="Concat", axis=1,
                inputs=[reg, obj, cls], outputs=[name],
            )
            oh.insert_after(model, oh.find_node(model, concat_name), new)
            oh.add_output(model, name, (1, depth, h, w))
            summary.append(f"{name}=(1,{depth},{h},{w})@s{stride}[reg={reg},obj={obj},cls={cls}]")

        log.info("yolox: exposed 3 tensors %s", "  ".join(summary))
        return model
