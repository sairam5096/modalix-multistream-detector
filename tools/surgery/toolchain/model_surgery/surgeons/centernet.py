"""
CenterNet (Objects-as-Points) detection-head surgery.

CenterNet is **not** a YOLO. It is a keypoint/heatmap detector with a *single*
output scale (output stride 4 — e.g. 512x512 input -> 128x128 maps) and **no
anchors, no FPN, no DFL**. The backbone (DLA-34 / ResNet / Hourglass) feeds one
shared feature map into three sibling conv branches ("head"):

    heatmap (hm)  : (1, nc, H/4, W/4)   per-class centre logits  (needs Sigmoid)
    wh (size)     : (1, 2,  H/4, W/4)   box w,h in output-pixel (stride-4) units
    reg (offset)  : (1, 2,  H/4, W/4)   sub-pixel centre offset  in [~0,1]

Each branch is `feat -> Conv3x3 -> ReLU -> Conv1x1(->D)` (D in {nc, 2, 2}); all
three consume the *same* shared feature tensor. Decode (done downstream by the
plugin, NOT here) is: sigmoid(hm) -> 3x3 max-pool "NMS" (keep local maxima) ->
top-k over all (class,y,x) -> per peak (cls,y,x):
    cx=(x+reg_x)*4  cy=(y+reg_y)*4  w=wh_w*4  h=wh_h*4  score=hm_prob.

This surgeon does **not** decode and does **not** sigmoid. It locates the three
head branches structurally and re-exposes their raw conv outputs as three
canonical graph outputs `heatmap`, `wh`, `reg` (each 1 scale, stride 4). Any
decode tail the export may carry (Sigmoid / MaxPool / TopK / Gather /
GatherElements) is left detached from the outputs and dropped by
`save_model`'s reachability prune — same idiom as the yolox / v10 surgeons.

Contract produced (differs from every YOLO family — see report notes):
    3 tensors, ONE scale, stride 4, heterogeneous depths [nc, 2, 2]:
        heatmap (1, nc, H/4, W/4)   raw logits   class_is_prob = false
        wh      (1, 2,  H/4, W/4)   raw
        reg     (1, 2,  H/4, W/4)   raw
    decode_type = "centernet".

Locating the head (naming-agnostic — CenterNet exports are often numeric):
  1. Candidate final head convs = Conv nodes whose output channel-depth is `nc`
     (heatmap, exactly one) or `2` (wh/reg, at least two).
  2. Shared feature `F` = the nearest tensor on the heatmap conv's input-chain
     that also lies on >=2 of the depth-2 convs' input-chains (the fork point).
  3. The head = the heatmap conv + the two depth-2 convs whose chains reach `F`.
     This rejects any stray depth-2 conv elsewhere in the backbone.

Splitting the two depth-2 branches into wh vs reg (both are structurally
identical) uses, in order:
  (a) name hints on the branch conv/weight/output names (`wh`/`size` -> wh;
      `reg`/`off`/`offset` -> reg) — fires on semantic exports (xingyizhou/mmdet
      when output names survive), no-ops on numeric exports;
  (b) a best-effort magnitude probe: one onnxruntime forward pass; wh spans the
      full [0, H/4] size range while reg is a sub-pixel ~[0,1] offset, so
      max|wh| >> max|reg| (observed ratio ~140x). wh = the larger-magnitude
      head. Skipped silently if onnxruntime is unavailable / the probe fails;
  (c) positional fallback `DEPTH2_ORDER` = (reg, wh) in graph-output / node
      order — the order of the validated SiMa `UR_onnx_centernet` export
      (hm, reg, wh). NOTE this order is export-dependent (canonical xingyizhou
      is hm, wh, reg), which is exactly why (a)/(b) take precedence.

Validated against SiMa's `UR_onnx_centernet_fp32_512_512.onnx` (DLA-ish
backbone, 512x512, 80 classes, stride-4 128x128 heads, tail-less export): the
locator finds head convs [hm=508, reg=511, wh=514] off shared feature 505, the
probe splits wh(max|.|~132) from reg(max|.|~1.0), and the surgered graph runs in
onnxruntime returning heatmap(1,80,128,128), wh(1,2,128,128), reg(1,2,128,128).
"""

from __future__ import annotations

import logging

import onnx
from onnx import shape_inference

from .. import onnx_helpers as oh
from ..identify import YoloIdentity
from .base import SurgeonBase

log = logging.getLogger("model_surgery.surgeon.centernet")

STRIDE = 4                       # single output scale
HEATMAP, WH, REG = "heatmap", "wh", "reg"
DEPTH2_ORDER = (REG, WH)         # positional fallback (SiMa UR export order: hm, reg, wh)

# substring hints (lower-cased) for name-based wh/reg disambiguation
_WH_HINTS = ("wh", "size", "_s.", "/s/")
_REG_HINTS = ("reg", "off", "offset", "hm_off", "ct_off")


def _shapes(model: onnx.ModelProto) -> dict[str, list[int | None]]:
    """name -> dim list (None for dynamic) for every typed tensor, shapes inferred."""
    vi: dict[str, list[int | None]] = {}
    for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        vi[v.name] = [
            d.dim_value if d.HasField("dim_value") else None
            for d in v.type.tensor_type.shape.dim
        ]
    return vi


def _producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {o: n for n in model.graph.node for o in n.output}


def _strip_initializer_inputs(model: onnx.ModelProto) -> int:
    """
    IR-3 hygiene: drop `graph.input` entries that are also initializers.

    Pre-DLA CenterNet exports (xingyizhou, opset 9 / IR 3) list every initializer
    as a graph input too. After surgery's dead-initializer prune, such an entry
    becomes a *dangling required input* with no backing value — onnxruntime then
    demands it as a feed. Removing the duplicates up front (the standard
    `remove_initializer_from_input` normalization) is always safe: the initializer
    still supplies the constant. No-op on modern (IR>=4) exports.
    """
    init = {i.name for i in model.graph.initializer}
    dup = [vi for vi in model.graph.input if vi.name in init]
    for vi in dup:
        model.graph.input.remove(vi)
    if dup:
        log.info("stripped %d initializer(s) duplicated as graph inputs (IR-3 export)", len(dup))
    return len(dup)


def _input_chain(prod: dict[str, onnx.NodeProto], tensor: str, maxhops: int = 12) -> list[str]:
    """Ordered tensors climbing input[0] from `tensor` (nearest first)."""
    chain: list[str] = []
    t = tensor
    for _ in range(maxhops):
        chain.append(t)
        n = prod.get(t)
        if n is None or not n.input:
            break
        t = n.input[0]
    return chain


def _name_role(node: onnx.NodeProto) -> str | None:
    """Guess wh/reg from a branch conv's node/weight/output names, else None."""
    toks = " ".join([node.name or "", node.output[0]] + list(node.input)).lower()
    if any(h in toks for h in _REG_HINTS):
        return REG
    if any(h in toks for h in _WH_HINTS):
        return WH
    return None


class SurgeonCenterNet(SurgeonBase):
    name = "centernet"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        H, W, nc = ident.height, ident.width, ident.num_classes
        log.info("centernet surgery (%dx%d, %d cls, single scale stride %d, heatmap head)",
                 W, H, nc, STRIDE)

        # normalize the common old-IR3 CenterNet export quirk before editing
        _strip_initializer_inputs(model)

        # CenterNet exports carry little/no intermediate value_info; infer shapes on
        # a copy so we can locate the head by output channel depth (names vary).
        try:
            inferred = shape_inference.infer_shapes(model)
        except Exception as exc:  # pragma: no cover - malformed model
            raise ValueError(f"shape inference failed; cannot locate CenterNet head: {exc}")
        vi = _shapes(inferred)
        prod = _producer_map(model)

        def out_depth(n: onnx.NodeProto) -> int | None:
            s = vi.get(n.output[0])
            return s[1] if s and len(s) >= 2 else None

        # 1) candidate final head convs: depth == nc (heatmap) or == 2 (wh/reg)
        hm_convs = [n for n in model.graph.node if n.op_type == "Conv" and out_depth(n) == nc]
        d2_convs = [n for n in model.graph.node if n.op_type == "Conv" and out_depth(n) == 2]
        if not hm_convs or len(d2_convs) < 2:
            raise ValueError(
                f"not a CenterNet head: found {len(hm_convs)} conv(s) of depth nc={nc} and "
                f"{len(d2_convs)} conv(s) of depth 2 (need 1 heatmap + 2 wh/reg sibling convs "
                f"at a single scale). Already surgeoned or wrong family?")

        # 2) shared feature F = nearest tensor on a heatmap conv's input-chain that
        #    also lies on >=2 depth-2 convs' chains (the head fork point).
        d2_chains = {id(n): set(_input_chain(prod, n.input[0])) for n in d2_convs}
        head_hm: onnx.NodeProto | None = None
        feat: str | None = None
        wh_reg: list[onnx.NodeProto] = []
        for hm in hm_convs:
            for t in _input_chain(prod, hm.input[0]):
                sharing = [n for n in d2_convs if t in d2_chains[id(n)]]
                if len(sharing) >= 2:
                    head_hm, feat, wh_reg = hm, t, sharing[:2]
                    break
            if head_hm is not None:
                break
        if head_hm is None:
            raise ValueError(
                "could not group a heatmap conv with two depth-2 sibling convs off a shared "
                "feature — head is not the expected CenterNet 3-branch (hm/wh/reg) structure")

        heatmap_out = head_hm.output[0]
        hm_depth = out_depth(head_hm)
        if hm_depth != nc:
            log.warning("heatmap conv depth %s != ident.num_classes %d", hm_depth, nc)
        log.info("located CenterNet head off shared feature '%s': heatmap=%s (nc=%s), "
                 "depth-2 branches=%s", feat, heatmap_out, hm_depth, [n.output[0] for n in wh_reg])

        # 3) split the two depth-2 branches into wh vs reg
        wh_out, reg_out = self._split_wh_reg(model, wh_reg)
        log.info("centernet head roles: heatmap=%s  wh=%s  reg=%s", heatmap_out, wh_out, reg_out)

        # 4) re-expose the three raw head tensors as canonical outputs. Identity
        #    renames without touching values (heatmap stays raw logits — the plugin
        #    applies Sigmoid). Clearing the old outputs first detaches any
        #    sigmoid/maxpool/topk/gather decode tail, which save_model then prunes.
        oh.clear_outputs(model)
        h4, w4 = H // STRIDE, W // STRIDE
        for canon, src, depth in ((HEATMAP, heatmap_out, nc), (WH, wh_out, 2), (REG, reg_out, 2)):
            ident_node = oh.make_node(
                name=f"/centernet/head/{canon}/Identity", op_type="Identity",
                inputs=[src], outputs=[canon],
            )
            oh.insert_after(model, oh.find_node_by_output(model, src), ident_node)
            oh.add_output(model, canon, (1, depth, h4, w4))

        log.info("centernet: exposed 3 tensors  heatmap=(1,%d,%d,%d)  wh=(1,2,%d,%d)  "
                 "reg=(1,2,%d,%d) @ stride %d", nc, h4, w4, h4, w4, h4, w4, STRIDE)
        return model

    # ----------------------------------------------------------------------- #
    def _split_wh_reg(self, model: onnx.ModelProto,
                      d2: list[onnx.NodeProto]) -> tuple[str, str]:
        """Return (wh_output_name, reg_output_name) for the two depth-2 head convs."""
        a, b = d2[0], d2[1]

        # (a) name hints (semantic exports)
        ra, rb = _name_role(a), _name_role(b)
        if ra and rb and ra != rb:
            wh = a.output[0] if ra == WH else b.output[0]
            reg = b.output[0] if ra == WH else a.output[0]
            log.info("wh/reg split by name hints")
            return wh, reg
        if ra and not rb:
            return (a.output[0], b.output[0]) if ra == WH else (b.output[0], a.output[0])
        if rb and not ra:
            return (b.output[0], a.output[0]) if rb == WH else (a.output[0], b.output[0])

        # (b) magnitude probe: wh spans ~[0, H/4], reg is a sub-pixel ~[0,1] offset,
        #     so max|wh| >> max|reg|. Best-effort; skipped on any failure.
        probed = self._probe_magnitudes(model, [a.output[0], b.output[0]])
        if probed is not None:
            (na, ma), (nb, mb) = probed
            hi, lo = max(ma, mb), min(ma, mb)
            if hi >= 3.0 * max(lo, 1e-6):     # decisive separation
                wh = na if ma >= mb else nb
                reg = nb if ma >= mb else na
                log.info("wh/reg split by magnitude probe (max|.|: %s=%.2f, %s=%.2f)",
                         na, ma, nb, mb)
                return wh, reg
            log.warning("magnitude probe inconclusive (%s=%.2f, %s=%.2f) — using positional fallback",
                        na, ma, nb, mb)

        # (c) positional fallback: DEPTH2_ORDER over graph-output / node order.
        ordered = self._order_by_position(model, d2)
        role_of = dict(zip(DEPTH2_ORDER, [n.output[0] for n in ordered]))
        log.warning("wh/reg split by positional fallback %s (export-dependent — confirm!)",
                    DEPTH2_ORDER)
        return role_of[WH], role_of[REG]

    @staticmethod
    def _order_by_position(model: onnx.ModelProto,
                           d2: list[onnx.NodeProto]) -> list[onnx.NodeProto]:
        """Order the two depth-2 convs by original graph-output order, else node order."""
        out_pos = {o.name: i for i, o in enumerate(model.graph.output)}
        node_pos = {id(n): i for i, n in enumerate(model.graph.node)}
        return sorted(d2, key=lambda n: (out_pos.get(n.output[0], 1 << 30), node_pos[id(n)]))

    @staticmethod
    def _probe_magnitudes(model: onnx.ModelProto,
                          tensors: list[str]) -> list[tuple[str, float]] | None:
        """One ORT forward pass; return [(name, max|value|), ...] or None on failure."""
        try:
            import numpy as np
            import onnxruntime as ort  # type: ignore

            inputs = oh.input_tensors(model)
            if not inputs:
                return None
            probe = onnx.ModelProto()
            probe.CopyFrom(model)
            # real elem_type per tensor (fp16 exports reject a FLOAT declaration)
            try:
                inferred = shape_inference.infer_shapes(probe)
                dtypes = {v.name: v.type.tensor_type.elem_type
                          for v in inferred.graph.value_info}
            except Exception:
                dtypes = {}
            del probe.graph.output[:]
            for t in tensors:
                et = dtypes.get(t) or onnx.TensorProto.FLOAT
                probe.graph.output.append(
                    onnx.helper.make_tensor_value_info(t, et, None))
            sess = ort.InferenceSession(probe.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
            feeds = {}
            for i in sess.get_inputs():
                dt = {"tensor(float16)": np.float16, "tensor(double)": np.float64}.get(
                    i.type, np.float32)
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
                feeds[i.name] = np.random.RandomState(0).rand(*shape).astype(dt)
            outs = sess.run(tensors, feeds)
            return [(t, float(np.abs(v).max())) for t, v in zip(tensors, outs)]
        except Exception as exc:
            log.info("magnitude probe unavailable (%s)", exc)
            return None
