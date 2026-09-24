"""
EfficientDet (anchor-based, BiFPN) detection-head surgery.

EfficientDet is a *new* architecture family for this tool — unrelated to YOLO.
Backbone EfficientNet -> BiFPN neck -> a **shared** class-net and box-net applied
to **five** pyramid levels P3..P7 (strides 8/16/32/64/128), with **9 anchors per
location** (3 scales x 3 aspect ratios). Per level the two head predictors emit

    class head : (1, na*nc, H/s, W/s)   raw class logits   (na=9 -> 810 for nc=90)
    box   head : (1, na*4,  H/s, W/s)   raw box regression (na=9 -> 36)

The box regression is anchor-relative (ty,tx,th,tw, decoded against each anchor's
center/size); the class output is sigmoid logits. Because the head convs are
*weight-shared across all five levels* (only the interleaved BatchNorm differs),
a single `class_net.predict.conv_pw.weight` / `box_net.predict.conv_pw.weight`
initializer is consumed by five Conv nodes — one per level. That shared-predictor
signature (plus the `fpn`/`class_net`/`box_net` namespaces and the 5-level
pyramid) is how EfficientDet is told apart from any YOLO head (no cv2/cv3/dfl,
no `m.i/Conv`, no decoupled `[4,1,nc]` concat).

Like the anchor-based YOLOv5/v7 path, surgery here does **not** decode. The SiMa
generic box-decoder / the nx_neat plugin owns the anchor+ty,tx,th,tw math and the
sigmoid + NMS. All this surgeon does is **re-expose the ten raw per-level head
convs** as the graph outputs

    cls_0..cls_4   (1, na*nc, H/s, W/s)     # stride 8,16,32,64,128 order (P3->P7)
    box_0..box_4   (1, na*4,  H/s, W/s)

and detach whatever decode/concat/NMS tail the export carried after the heads
(raw exports have none; "concat form" exports reshape/permute/concat to
(1, N, nc)+(1, N, 4); AutoML exports add anchors+NMS). We re-point the graph
outputs at the ten conv tensors and drop the model's own outputs, so the entire
tail becomes unreachable and `save_model()`'s reachability prune deletes it — no
explicit tail enumeration, exactly like the YOLOv5 surgeon (5 levels x 2 heads
instead of 3 raw convs).

The five weight-shared convs are indistinguishable by weight, so they are ordered
into P3..P7 by their inferred output spatial size (largest area == stride 8),
falling back to graph emission order (effdet emits P3..P7) when a level's size is
left symbolic by the BiFPN Resize ops. The class and box heads are ordered
independently and paired by level index.

Contract produced (10 tensors, raw logits — anchor-based, `class_is_prob=false`):

    cls_0 : (1, 810, 64, 64)   box_0 : (1, 36, 64, 64)   # stride 8   (P3)
    cls_1 : (1, 810, 32, 32)   box_1 : (1, 36, 32, 32)   # stride 16  (P4)
    cls_2 : (1, 810, 16, 16)   box_2 : (1, 36, 16, 16)   # stride 32  (P5)
    cls_3 : (1, 810,  8,  8)   box_3 : (1, 36,  8,  8)   # stride 64  (P6)
    cls_4 : (1, 810,  4,  4)   box_4 : (1, 36,  4,  4)   # stride 128 (P7)
                                                          # (D0, 512x512, nc=90, na=9)

Validated against a real EfficientDet-D0 graph exported from `effdet`
(rwightman) at 512x512 — the raw-heads export and a concat-form export — see the
module surgery notes / the agent report. NOTE: this family needs 5-level support
that the shared `contract.py` (STRIDES=(8,16,32), 3-tensor anchor-based spec)
does not yet provide; the surgeon is self-contained and does not depend on
`contract.py` for the level count. The CLI/metadata/JSON wiring changes required
to run this end-to-end through `python -m model_surgery` are listed in the report.
"""

from __future__ import annotations

import logging

import onnx
from onnx import helper as onnx_helper

from .. import onnx_helpers as oh
from ..identify import YoloIdentity
from .base import SurgeonBase

log = logging.getLogger("model_surgery.surgeon.efficientdet")

# EfficientDet detection pyramid: P3..P7, five levels (cf. YOLO's 3).
EFFDET_STRIDES = (8, 16, 32, 64, 128)

# Canonical anchor-based contract output prefixes (per-level, P3..P7 order).
CLS_PREFIX = "cls_"
BOX_PREFIX = "box_"

# Weight-initializer tags that identify the shared class / box predictor convs.
# Match the *pointwise* predictor (1x1) of the head's final SeparableConv2d; the
# `net.`/`model.`/`` module prefix is absorbed by the substring test.
CLS_PREDICT_TAG = "class_net.predict"
BOX_PREDICT_TAG = "box_net.predict"


class SurgeonEfficientDet(SurgeonBase):
    """Re-expose the ten raw per-level EfficientDet head convs; strip the tail."""

    name = "efficientdet"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        H, W, nc = ident.height, ident.width, ident.num_classes
        strides = EFFDET_STRIDES
        n_levels = len(strides)
        log.info("efficientdet surgery (%dx%d, %d cls, %d levels P3..P7, "
                 "anchor-based BiFPN head) — expose %d raw head convs",
                 W, H, nc, n_levels, 2 * n_levels)

        # A valid topological node order (a) lets data-propagation shape inference
        # resolve the per-level map sizes we order by, and (b) makes the collected
        # convs' graph order a true P3..P7 emission order for the fallback. Real
        # exports are already topo-ordered so this is a no-op; save_model re-sorts
        # regardless, so reordering here is semantically harmless.
        oh.topo_sort(model)

        # 1) locate the five shared class + five shared box predictor convs.
        cls_convs = self._predict_convs(model, CLS_PREDICT_TAG, n_levels, "class")
        box_convs = self._predict_convs(model, BOX_PREDICT_TAG, n_levels, "box")

        # 2) read head depths + dtype; derive/verify the anchor count.
        cls_w = oh.find_initializer_value(model, cls_convs[0].input[1])   # (na*nc,C,1,1)
        box_w = oh.find_initializer_value(model, box_convs[0].input[1])   # (na*4, C,1,1)
        cls_depth, box_depth = int(cls_w.shape[0]), int(box_w.shape[0])
        if box_depth % 4 != 0:
            raise ValueError(
                f"EfficientDet box head depth {box_depth} is not a multiple of 4 "
                f"(expected na*4, e.g. 36 for na=9) — not a standard box-net predictor")
        na = box_depth // 4
        if cls_depth != na * nc:
            raise ValueError(
                f"EfficientDet class head depth {cls_depth} != num_anchors*num_classes "
                f"= {na}*{nc} = {na * nc} (box head implies na={na}); check num_classes")
        elem_type = onnx_helper.np_dtype_to_tensor_dtype(cls_w.dtype)
        log.info("  na=%d (box depth %d), nc=%d (class depth %d), dtype=%s",
                 na, box_depth, nc, cls_depth, cls_w.dtype)

        # 3) order both head groups into P3..P7 (stride 8..128) — weight-shared
        #    convs are told apart only by output spatial size.
        cls_convs = self._order_by_level(model, cls_convs, H, W, strides, "class")
        box_convs = self._order_by_level(model, box_convs, H, W, strides, "box")

        # 4) re-point the graph outputs at the ten conv tensors (cls block, then
        #    box block). The model's own decode/concat/NMS tail — if any — then
        #    feeds no output and is pruned by save_model()'s reachability sweep.
        oh.clear_outputs(model)
        summary = []
        for i, stride in enumerate(strides):
            h, w = H // stride, W // stride
            self._expose(model, cls_convs[i], f"{CLS_PREFIX}{i}", (1, cls_depth, h, w), elem_type)
            summary.append(f"{CLS_PREFIX}{i}=(1,{cls_depth},{h},{w})@s{stride}")
        for i, stride in enumerate(strides):
            h, w = H // stride, W // stride
            self._expose(model, box_convs[i], f"{BOX_PREFIX}{i}", (1, box_depth, h, w), elem_type)
            summary.append(f"{BOX_PREFIX}{i}=(1,{box_depth},{h},{w})@s{stride}")
        log.info("efficientdet: exposed %d tensors  %s", 2 * n_levels, "  ".join(summary))
        return model

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _predict_convs(model: onnx.ModelProto, tag: str, n_levels: int,
                       which: str) -> list[onnx.NodeProto]:
        """The `n_levels` weight-shared pointwise predictor convs for one head.

        A stock EfficientDet head applies one shared SeparableConv2d predictor to
        every pyramid level, so the pointwise (1x1) conv weight `*.{tag}.*.weight`
        is consumed by exactly `n_levels` Conv nodes. Match the pointwise (skip the
        3x3 depthwise) so the group is the per-level class/box logit producers.
        """
        convs: list[onnx.NodeProto] = []
        for n in model.graph.node:
            if n.op_type != "Conv" or len(n.input) < 2:
                continue
            wname = n.input[1]
            if tag not in wname or not wname.endswith(".weight"):
                continue
            try:
                w = oh.find_initializer_value(model, wname)
            except KeyError:
                continue
            if w.ndim == 4 and w.shape[2] == 1 and w.shape[3] == 1:  # pointwise predictor
                convs.append(n)
        if len(convs) != n_levels:
            raise ValueError(
                f"expected {n_levels} weight-shared EfficientDet {which} predictor convs "
                f"(pointwise '*{tag}*.weight'), found {len(convs)} — not a standard "
                f"EfficientDet {n_levels}-level (P3..P7) head (already surgeoned, a "
                f"non-{n_levels}-level variant, or a differently-named export?)")
        return convs

    @staticmethod
    def _infer_spatial(model: onnx.ModelProto,
                       names: set[str]) -> dict[str, tuple[int, int]]:
        """name -> (H,W) for the requested tensors, via data-propagation shape
        inference. BiFPN Resize ops can leave some levels symbolic; those are
        simply absent from the returned map."""
        try:
            mi = onnx.shape_inference.infer_shapes(model, data_prop=True)
        except Exception as exc:  # pragma: no cover - malformed model
            log.debug("efficientdet: shape inference failed (%s); using graph order", exc)
            return {}
        out: dict[str, tuple[int, int]] = {}
        for v in list(mi.graph.value_info) + list(mi.graph.output):
            if v.name not in names:
                continue
            dims = v.type.tensor_type.shape.dim
            if (len(dims) == 4 and dims[2].HasField("dim_value") and dims[3].HasField("dim_value")
                    and dims[2].dim_value > 0 and dims[3].dim_value > 0):
                out[v.name] = (dims[2].dim_value, dims[3].dim_value)
        return out

    def _order_by_level(self, model: onnx.ModelProto, convs: list[onnx.NodeProto],
                        H: int, W: int, strides: tuple[int, ...],
                        which: str) -> list[onnx.NodeProto]:
        """Sort `convs` into P3..P7 (stride 8..128) order.

        Each level's expected map size (H//s, W//s) is distinct, so a conv whose
        inferred output size matches a stride is placed in that slot. Convs whose
        size the BiFPN Resize left symbolic fall into the remaining slots in graph
        emission order (effdet emits P3..P7) — deterministic because each such conv
        maps to a unique unfilled slot.
        """
        expected = [(H // s, W // s) for s in strides]
        size_to_idx = {hw: i for i, hw in enumerate(expected)}
        shapes = self._infer_spatial(model, {c.output[0] for c in convs})

        ordered: list[onnx.NodeProto | None] = [None] * len(strides)
        leftover: list[onnx.NodeProto] = []
        for c in convs:                                   # convs are in graph order
            idx = size_to_idx.get(shapes.get(c.output[0]))
            if idx is not None and ordered[idx] is None:
                ordered[idx] = c
            else:
                leftover.append(c)
        empty = [i for i, o in enumerate(ordered) if o is None]
        for c, idx in zip(leftover, empty):               # graph order -> unfilled slots
            ordered[idx] = c

        if any(o is None for o in ordered) or len(leftover) != len(empty):
            log.warning("efficientdet: could not resolve all %s level sizes; "
                        "falling back to graph emission order (assumed P3..P7)", which)
            return list(convs)
        if leftover:
            log.info("efficientdet: %d %s level(s) had symbolic size; placed by graph order",
                     len(leftover), which)
        return ordered  # type: ignore[return-value]

    @staticmethod
    def _expose(model: onnx.ModelProto, conv: onnx.NodeProto, new_name: str,
                shape: tuple[int, ...], elem_type: int) -> None:
        """Rename a head conv's output to the canonical contract name, rewire every
        consumer (the decode tail) so the graph stays well-formed, and declare it a
        graph output. The rewired tail is unreachable from the new outputs and is
        pruned by save_model(). Output dtype is patched to the conv's own type so
        fp16 exports pass onnx.checker / onnxruntime (cf. the YOLOv5 surgeon)."""
        old_name = conv.output[0]
        conv.output[0] = new_name
        for node in model.graph.node:
            for j, inp in enumerate(node.input):
                if inp == old_name:
                    node.input[j] = new_name
        oh.add_output(model, new_name, shape)
        model.graph.output[-1].type.tensor_type.elem_type = elem_type
