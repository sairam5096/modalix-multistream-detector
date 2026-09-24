"""
Auto-identify a YOLO detection ONNX across families and dispatch the right
surgeon — the "identify the provided model" half of the tool.

Detectors, tried most-specific first:
  1. Meituan YOLOv6      — `detect.cls_preds.*` initializers
  2. Ultralytics anchor-free (v8/v9/v10/v11/yolo26) — `cv2.2/cv2.2.2/Conv` head
     (or `one2one_` variant), sub-routed by DFL / one2one / attention
  3. YOLOX               — decoupled head: 3 axis-1 Concats of depth [4,1,nc]
  4. Anchor-based v5/v7  — three `<head>/m.{i}/Conv` head convs

`family` is the canonical key (== surgeon_key == decode_type). v9 vs v8 and v5 vs
v7 are numerically identical surgeries; when a graph can't distinguish them we
pick one and the user can override with --variant.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import numpy_helper

from . import onnx_helpers as oh

log = logging.getLogger("model_surgery.identify")

# families that are NOT on the 6-tensor decoded-in-graph anchor-free YOLO path
# (drives the `anchor_free` flag -> box-decoder routing / class_is_prob)
ANCHOR_BASED_FAMILIES = {"yolov5", "yolov7", "efficientdet", "centernet"}
YOLOX_FAMILY = "yolox"

# YOLOv5 vs YOLOv7 export the same head structure but use DIFFERENT COCO anchor priors —
# the only signal that distinguishes them. We read the anchor_grid from the graph and match.
_V5_ANCHOR_SET = {10, 13, 16, 30, 33, 23, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326}
_V7_ANCHOR_SET = {12, 16, 19, 36, 40, 28, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401}


def _const_tensors(model: onnx.ModelProto):
    """Yield every constant tensor (initializers + Constant-node values) as np arrays."""
    for i in model.graph.initializer:
        yield numpy_helper.to_array(i)
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    yield numpy_helper.to_array(a.t)


def read_anchors(model: onnx.ModelProto) -> list | None:
    """Extract the 3-level [3,3,2] anchor priors from a v5/v7-style graph.

    The anchor_grid is exported as per-level `[1,3,H,W,2]` constants (each level's 3 anchors
    broadcast across the grid); the distinct anchor pair sits at [0,:,0,0,:]. Returns anchors
    ordered P3→P5 (largest grid first), or None if not found (e.g. anchors constant-folded away).
    """
    grids = []
    for arr in _const_tensors(model):
        a = np.asarray(arr, dtype=float)
        s = a.shape
        if len(s) == 5 and s[1] == 3 and s[4] == 2:
            uniq = np.unique(a.flatten())
            an = uniq[(uniq >= 6) & (uniq <= 1024)]
            if 0 < an.size <= 8:  # anchor_grid: few distinct anchor magnitudes per level
                grids.append((s[2], np.round(a[0, :, 0, 0, :]).astype(int).tolist()))
    grids.sort(key=lambda g: -g[0])  # P3 (80) → P4 (40) → P5 (20)
    return [g[1] for g in grids[:3]] if len(grids) >= 3 else None


def anchor_based_family(anchors: list | None) -> str | None:
    """Match extracted anchors to the YOLOv5 vs YOLOv7 COCO prior sets; None if inconclusive."""
    if not anchors:
        return None
    vals = {int(v) for lvl in anchors for wh in lvl for v in wh}
    o5 = len(vals & _V5_ANCHOR_SET) / len(_V5_ANCHOR_SET)
    o7 = len(vals & _V7_ANCHOR_SET) / len(_V7_ANCHOR_SET)
    if max(o5, o7) < 0.6:
        return None
    return "yolov7" if o7 > o5 else "yolov5"


@dataclass
class YoloIdentity:
    family: str                  # yolov5/6/7/8/9/10/11 | yolo26 | yolox
    height: int
    width: int
    num_classes: int
    head_prefix: str = ""        # "/model.22", "detect", or "" (yolox: structural)
    head_index: int = -1
    one2one_prefix: str = ""     # "" or "one2one_"
    dfl_weight: str = ""         # "" if no DFL
    has_dfl: bool = False
    has_attention: bool = False
    num_anchors: int = 1         # v5/v7 = 3
    num_kpts: int = 0            # pose only
    anchors: list | None = None  # v5/v7 [3,3,2] priors read from the graph (None = use defaults)
    flavor: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def surgeon_key(self) -> str:
        return self.family

    @property
    def decode_type(self) -> str:
        return self.family

    @property
    def anchor_free(self) -> bool:
        return self.family not in ANCHOR_BASED_FAMILIES

    @property
    def is_yolox(self) -> bool:
        return self.family == YOLOX_FAMILY

    @property
    def version(self) -> int:
        m = re.search(r"(\d+)", self.family)
        return int(m.group(1)) if m else 0

    @property
    def task(self) -> str:
        """Ultralytics task derived from the family suffix."""
        for suffix, t in (("-cls", "classify"), ("-seg", "segment"),
                          ("-pose", "pose"), ("-obb", "obb"), ("depth", "depth")):
            if self.family.endswith(suffix):
                return t
        return "detect"

    def describe(self) -> str:
        fl = f"-{self.flavor}" if self.flavor else ""
        tags = []
        if self.has_dfl: tags.append("DFL")
        elif self.family not in ANCHOR_BASED_FAMILIES and self.family != YOLOX_FAMILY: tags.append("no-DFL")
        if self.one2one_prefix: tags.append("one2one")
        if self.num_anchors > 1: tags.append(f"na={self.num_anchors}")
        tag = ("  " + "/".join(tags)) if tags else ""
        return (f"{self.family}{fl}  {self.width}x{self.height}  {self.num_classes} classes  "
                f"head={self.head_prefix or '(structural)'}{tag}")


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _find_prefix(model: onnx.ModelProto, substring: str) -> str | None:
    for node in model.graph.node:
        if substring in node.name:
            parts = node.name.split("/")
            return "/".join(parts[:3]) if len(parts) >= 3 else node.name
    return None


def _has_node_substr(model: onnx.ModelProto, substring: str) -> bool:
    return any(substring in n.name for n in model.graph.node)


def _input_hw(model: onnx.ModelProto) -> tuple[int, int]:
    inputs = oh.input_tensors(model)
    if not inputs:
        raise ValueError("no valid (non-initializer) input tensor found")
    shape = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
    if len(shape) != 4:
        raise ValueError(f"expected 4D NCHW input, got shape {shape}")
    _, _, H, W = shape
    if H <= 0 or W <= 0:
        raise ValueError(f"input H/W must be static; got H={H} W={W}")
    return H, W


def _init_names(model: onnx.ModelProto) -> set[str]:
    return {i.name for i in model.graph.initializer}


# --------------------------------------------------------------------------- #
# 1) Meituan YOLOv6
# --------------------------------------------------------------------------- #
def _id_v6(model: onnx.ModelProto, H: int, W: int) -> YoloIdentity | None:
    inits = _init_names(model)
    if "detect.cls_preds.0.weight" not in inits:
        return None
    nc = int(oh.find_initializer_value(model, "detect.cls_preds.0.weight").shape[0])
    proj = "detect.proj_conv.weight"
    has_dfl = proj in inits
    return YoloIdentity(family="yolov6", height=H, width=W, num_classes=nc,
                        head_prefix="detect", dfl_weight=proj if has_dfl else "",
                        has_dfl=has_dfl)


# --------------------------------------------------------------------------- #
# 2) Ultralytics anchor-free (v8 / v9 / v10 / v11 / yolo26)
# --------------------------------------------------------------------------- #
def _id_anchorfree(model: onnx.ModelProto, H: int, W: int) -> YoloIdentity | None:
    one2one = ""
    prefix = _find_prefix(model, "cv2.2/cv2.2.2/Conv")
    if not prefix:
        prefix = _find_prefix(model, "one2one_cv2.2/one2one_cv2.2.2/Conv")
        one2one = "one2one_"
    if not prefix:
        return None
    if "cv2.2" in prefix:
        prefix = prefix.removesuffix(f"/{one2one}cv2.2")

    nc = int(oh.find_initializer_value(model, f"{prefix[1:]}.{one2one}cv3.2.2.weight").shape[0])
    has_dfl = any(i.name.endswith(".dfl.conv.weight") for i in model.graph.initializer)
    has_attn = _has_node_substr(model, "attn/qkv/conv/Conv")

    if not has_dfl:
        family = "yolo26"                               # one2one, DFL removed
    elif one2one:
        family = "yolov10"                              # one2one + DFL
    elif _find_prefix(model, "m/m.0/attn/qkv/conv/Conv"):
        family = "yolov11"                              # C2PSA attention
    elif _has_node_substr(model, "/cv5/"):
        family = "yolov9"                               # GELAN cv5 (harmless if missed → v8)
    else:
        family = "yolov8"

    try:
        head_index = int(prefix.split("/model.")[1].split("/")[0])
    except (IndexError, ValueError):
        head_index = -1

    # task sub-classification: seg/pose/obb all carry an extra `cv4` head off the
    # v8 Detect base. They differ by that head's depth (+ a proto branch for seg):
    #   seg  -> cv4 = 32 mask coeffs  AND a `/proto/` branch
    #   obb  -> cv4 = 1  (the oriented-box angle, Ultralytics `ne == 1`)
    #   pose -> cv4 = 3*num_kpts  (COCO: 51 = 3*17)
    num_kpts = 0
    # yolo26-pose keypoints live in a SEPARATE `{o}cv4_kpts.{i}` conv (3*num_kpts), not `cv4.{i}.2`.
    kpts_w = f"{prefix[1:]}.{one2one}cv4_kpts.2.weight"
    cv4_w = f"{prefix[1:]}.{one2one}cv4.2.2.weight"
    if oh.is_initializer(model, kpts_w):
        num_kpts = int(oh.find_initializer_value(model, kpts_w).shape[0]) // 3
        family = f"{family}-pose"
    elif oh.is_initializer(model, cv4_w):
        cv4_depth = int(oh.find_initializer_value(model, cv4_w).shape[0])
        if _has_node_substr(model, "/proto/"):
            family = f"{family}-seg"                 # mask coeffs (32) + proto
        elif cv4_depth == 1:
            family = f"{family}-obb"                 # single angle channel
        else:
            family = f"{family}-pose"                # keypoints (3*num_kpts)
            num_kpts = cv4_depth // 3

    return YoloIdentity(
        family=family, height=H, width=W, num_classes=nc,
        head_prefix=prefix, head_index=head_index, one2one_prefix=one2one,
        dfl_weight=f"{prefix[1:]}.dfl.conv.weight" if has_dfl else "",
        has_dfl=has_dfl, has_attention=has_attn, num_kpts=num_kpts,
    )


# --------------------------------------------------------------------------- #
# 3) YOLOX (decoupled anchor-free head)
# --------------------------------------------------------------------------- #
def _id_yolox(model: onnx.ModelProto, H: int, W: int) -> YoloIdentity | None:
    try:
        m = onnx.shape_inference.infer_shapes(model)
    except Exception:
        return None
    vi = {v.name: [d.dim_value if d.HasField("dim_value") else None
                   for d in v.type.tensor_type.shape.dim]
          for v in list(m.graph.value_info) + list(m.graph.output)}

    def depth(t):
        s = vi.get(t); return s[1] if s and len(s) >= 2 else None

    ncs = []
    for n in m.graph.node:
        if n.op_type != "Concat" or len(n.input) != 3:
            continue
        if next((a.i for a in n.attribute if a.name == "axis"), None) != 1:
            continue
        osh = vi.get(n.output[0])
        din = [depth(t) for t in n.input]
        if osh and len(osh) == 4 and din[0] == 4 and din[1] == 1 and din[2] and osh[1] == 5 + din[2]:
            ncs.append(din[2])
    if len(ncs) < 3 or len(set(ncs)) != 1:
        return None
    return YoloIdentity(family="yolox", height=H, width=W, num_classes=ncs[0],
                        head_prefix="", head_index=-1)


# --------------------------------------------------------------------------- #
# 4) Anchor-based v5 / v7
# --------------------------------------------------------------------------- #
def _id_anchor_based(model: onnx.ModelProto, H: int, W: int) -> YoloIdentity | None:
    # group `<prefix>/m.<i>/Conv` head convs (Detect/IDetect), i in {0,1,2}
    pat = re.compile(r"^(.*)/m\.(\d+)/Conv$")
    groups: dict[str, dict[int, onnx.NodeProto]] = {}
    for n in model.graph.node:
        if n.op_type != "Conv":
            continue
        mo = pat.match(n.name)
        if mo:
            groups.setdefault(mo.group(1), {})[int(mo.group(2))] = n

    for head, convs in groups.items():
        if not {0, 1, 2} <= set(convs):
            continue
        depths = {int(oh.find_initializer_value(model, convs[i].input[1]).shape[0]) for i in (0, 1, 2)}
        if len(depths) != 1:
            continue
        D = depths.pop()
        # nc from a rank-3 decode output (1, A, 5+nc) if present, else assume na=3
        nc = None
        for o in model.graph.output:
            s = [d.dim_value for d in o.type.tensor_type.shape.dim]
            if len(s) == 3 and s[2] > 5:
                nc = s[2] - 5
                break
        na = 3 if nc is None else max(1, D // (5 + nc))
        if nc is None:
            nc = D // na - 5
        if D % (5 + nc) != 0:
            continue
        # v5 and v7 share this head; distinguish by the anchor priors read from the graph.
        anchors = read_anchors(model)
        family = anchor_based_family(anchors) or "yolov5"   # default v5 if anchors folded away
        if anchors is None:
            log.info("anchor grid not recoverable; defaulting family=yolov5 (override with --variant yolov7)")
        else:
            log.info("anchor-based family=%s (from graph anchor priors)", family)
        return YoloIdentity(family=family, height=H, width=W, num_classes=nc,
                            head_prefix=head, num_anchors=D // (5 + nc), anchors=anchors)
    return None


# --------------------------------------------------------------------------- #
# 5) EfficientDet (anchor-based, BiFPN, 5 pyramid levels) — non-YOLO
# --------------------------------------------------------------------------- #
def _id_efficientdet(model: onnx.ModelProto, H: int, W: int) -> YoloIdentity | None:
    consumed = {inp for n in model.graph.node if n.op_type == "Conv" for inp in n.input}

    def predictor(tag: str) -> str | None:
        for w in _init_names(model):
            if tag in w and w.endswith(".weight"):
                convs = [n for n in model.graph.node
                         if n.op_type == "Conv" and len(n.input) > 1 and n.input[1] == w]
                # the head PREDICT conv is shared across 5 levels AND terminal
                # (its output feeds reshape/output, not another Conv — unlike the
                # intermediate separable feature convs).
                if len(convs) == 5 and all(c.output[0] not in consumed for c in convs):
                    return w
        return None

    cw, bw = predictor("class_net.predict"), predictor("box_net.predict")
    if not cw or not bw:
        return None
    cls_depth = int(oh.find_initializer_value(model, cw).shape[0])
    box_depth = int(oh.find_initializer_value(model, bw).shape[0])
    if box_depth % 4 or cls_depth % (box_depth // 4):
        return None
    na = box_depth // 4
    return YoloIdentity(family="efficientdet", height=H, width=W,
                        num_classes=cls_depth // na, num_anchors=na)


# --------------------------------------------------------------------------- #
# 6) CenterNet (Objects-as-Points) — single scale (stride 4), heatmap+wh+reg
# --------------------------------------------------------------------------- #
def _id_centernet(model: onnx.ModelProto, H: int, W: int) -> YoloIdentity | None:
    try:
        m = onnx.shape_inference.infer_shapes(model)
    except Exception:
        return None

    def dims(v):
        return [d.dim_value if d.HasField("dim_value") else None
                for d in v.type.tensor_type.shape.dim]

    qh, qw = H // 4, W // 4

    # primary: the 3 heads are exposed as graph outputs at H/4 with depths {nc,2,2}
    at_q = [s[1] for s in (dims(o) for o in m.graph.output)
            if len(s) == 4 and s[2] == qh and s[3] == qw]
    heat = [d for d in at_q if d and d != 2]
    if at_q.count(2) >= 2 and len(heat) == 1:
        return YoloIdentity(family="centernet", height=H, width=W, num_classes=heat[0])

    # fallback (decode-tail exports): a terminal Conv at H/4 that feeds a Sigmoid
    # is the heatmap; two sibling depth-2 terminal convs are wh/reg.
    vi = {v.name: dims(v) for v in list(m.graph.value_info) + list(m.graph.output)}
    consumed = {inp for n in m.graph.node if n.op_type == "Conv" for inp in n.input}
    to_sigmoid = {n.input[0] for n in m.graph.node if n.op_type == "Sigmoid"}
    depth2, heat2 = 0, None
    for n in m.graph.node:
        if n.op_type != "Conv" or n.output[0] in consumed:
            continue
        s = vi.get(n.output[0])
        if not s or len(s) != 4 or s[2] != qh or s[3] != qw:
            continue
        if s[1] == 2:
            depth2 += 1
        elif s[1] and s[1] != 2 and n.output[0] in to_sigmoid and heat2 is None:
            heat2 = s[1]
    if depth2 >= 2 and heat2:
        return YoloIdentity(family="centernet", height=H, width=W, num_classes=heat2)
    return None


# --------------------------------------------------------------------------- #
# 7) Ultralytics classification (image classifier — no detection head)
# --------------------------------------------------------------------------- #
def _id_classify(model: onnx.ModelProto, H: int, W: int) -> YoloIdentity | None:
    """An Ultralytics `-cls` model has NO detection head: the backbone feeds a
    `GlobalAveragePool -> Flatten -> Gemm (-> Softmax)` classifier emitting a
    single rank-2 `(1, nc)` score vector. Detected last, once every detection
    probe has declined."""
    outs = list(model.graph.output)
    if len(outs) != 1:
        return None
    dims = [d.dim_value for d in outs[0].type.tensor_type.shape.dim]
    if len(dims) != 2:                                    # (N, nc) only
        return None
    optypes = {n.op_type for n in model.graph.node}
    if "GlobalAveragePool" not in optypes or "Gemm" not in optypes:
        return None
    nc = dims[1]
    if nc <= 0:                                           # recover from the Gemm weight
        gemm = next(n for n in model.graph.node if n.op_type == "Gemm")
        nc = int(oh.find_initializer_value(model, gemm.input[1]).shape[0])
    # base family from the backbone signature (surgery is version-independent;
    # this only labels metadata). C2PSA attention -> v11-family; else v8-family.
    base = "yolov11" if _has_node_substr(model, "attn/qkv") else "yolov8"
    return YoloIdentity(family=f"{base}-cls", height=H, width=W, num_classes=nc,
                        head_prefix="", head_index=-1)


def _id_depth(model: onnx.ModelProto, H: int, W: int) -> YoloIdentity | None:
    """Depth-estimation model: a single DENSE output (1, 1, h, w) and no detection head.
    Surgery is op-compat only (attention lowering); the dense map is kept as-is."""
    outs = model.graph.output
    if len(outs) != 1:
        return None
    dims = [d.dim_value for d in outs[0].type.tensor_type.shape.dim]
    if len(dims) == 4 and dims[1] == 1 and dims[2] >= 64 and dims[3] >= 64:
        return YoloIdentity(family="depth", height=H, width=W, num_classes=0,
                            head_prefix="", head_index=-1)
    return None


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def identify(model_or_path, *, simplify: bool = True) -> tuple[onnx.ModelProto, YoloIdentity]:
    model = model_or_path if isinstance(model_or_path, onnx.ModelProto) \
        else oh.load_model(str(model_or_path), simplify=simplify)
    H, W = _input_hw(model)

    for detector in (_id_v6, _id_anchorfree, _id_yolox, _id_efficientdet,
                     _id_anchor_based, _id_centernet, _id_classify, _id_depth):
        ident = detector(model, H, W)
        if ident is not None:
            log.info("identified: %s", ident.describe())
            return model, ident

    raise ValueError("could not identify a supported YOLO model "
                     "(tried v6 / anchor-free v8-v11+yolo26 / yolox / anchor-based v5-v7 / "
                     "classify)")
