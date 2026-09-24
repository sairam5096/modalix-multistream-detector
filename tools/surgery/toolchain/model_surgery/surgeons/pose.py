"""
YOLOv8 / v11 pose-head surgery.

An Ultralytics pose model is the ordinary v8 DFL *detection* base — box branch
`cv2.{i}` (64ch DFL) + class branch `cv3.{i}` (nc=1 "person") — with one extra
head per scale: `cv4.{i}`, a `3*num_kpts` channel Conv emitting the *raw*
keypoint predictions. The SiMa generic box-decoder (`decode_type=yolov8-pose`)
does the keypoint decode itself, so surgery is exactly the v8 bbox+class rewrite
(reused verbatim from `SurgeonYoloV8`) PLUS re-exposing the three raw `cv4.{i}`
convs as `kpt_{0,1,2}`. The extra head is NOT decoded in-graph.

Contract — 9 tensors, strides 8/16/32, input HxW:
    bbox_{i}        (1, 4,           H/s, W/s)  decoded cx,cy,w,h (pixel space)
    class_prob_{i}  (1, nc,          H/s, W/s)  sigmoid           (nc=1 person)
    kpt_{i}         (1, 3*num_kpts,  H/s, W/s)  RAW cv4 keypoint head

Box-decoder reconstruction of keypoint j at grid cell (col,row), stride s:
    x = (2*kx + col) * s ,  y = (2*ky + row) * s ,  v = sigmoid(kv)
(no sigmoid on x,y — matches Ultralytics `Pose.kpts_decode`; anchor = col+0.5,
so `2*kx + col == 2*kx + (anchor-0.5)`).

Validated (Ultralytics 8.4.95) against yolov8n-pose (/model.22) and yolo11n-pose
(/model.23): 9 outputs, correct shapes, finite, kpt_i bit-identical to raw cv4.
"""

from __future__ import annotations

import logging

import onnx

from .. import contract, onnx_helpers as oh
from ..identify import YoloIdentity
from .base import SurgeonBase
from .yolov8 import SurgeonYoloV8, _LTRB_TO_XYWH, _grid_offset  # v8 box math, reused
from .yolo26 import SurgeonYolo26  # no-DFL/one2one base for yolo26-pose

log = logging.getLogger("model_surgery.surgeon.pose")


# --------------------------------------------------------------------------- #
# shared detection-base helpers (imported by seg.py too — both task heads sit
# on the identical v8 Detect base, so the bbox+class rewrite lives here once)
# --------------------------------------------------------------------------- #
def _rewrite_detection(model: onnx.ModelProto, ident: YoloIdentity) -> None:
    """Rewrite the v8 Detect base -> bbox_{i} (decoded cxcywh) + class_prob_{i}
    (sigmoid), declaring those 6 contract outputs.

    DFL heads (the standard pose/seg export) reuse `SurgeonYoloV8` verbatim. The
    rare no-DFL depth-4 box branch is decoded directly with the same
    LTRB->xywh + grid/stride math (imported from .yolov8), mirroring yolo26.
    """
    if ident.has_dfl and ident.dfl_weight and oh.is_initializer(model, ident.dfl_weight):
        SurgeonYoloV8().do_surgery(model, ident)
        return

    head, nc = ident.head_prefix, ident.num_classes
    H, W = ident.height, ident.width
    log.info("no-DFL box branch — direct LTRB decode on %s", head)
    oh.clear_outputs(model)
    for name, shape in contract.box_decoder_outputs(H, W, nc):
        oh.add_output(model, name, shape)

    for i, stride in enumerate(contract.STRIDES):
        cur_h, cur_w = H // stride, W // stride
        box_conv = oh.find_node(model, f"{head}/cv2.{i}/cv2.{i}.2/Conv")

        cname = f"{head}/decode/{i}/Conv"
        oh.add_initializer(model, f"{cname}.weight", _LTRB_TO_XYWH * stride)
        conv = oh.make_node(
            name=cname, op_type="Conv",
            inputs=[box_conv.output[0], f"{cname}.weight"], outputs=[f"{cname}_output"],
        )
        oh.insert_after(model, box_conv, conv)

        aname = f"{head}/decode/{i}/Add"
        oh.add_initializer(model, f"{aname}.Const", _grid_offset(cur_h, cur_w, stride))
        add = oh.make_node(
            name=aname, op_type="Add",
            inputs=[conv.output[0], f"{aname}.Const"], outputs=[f"bbox_{i}"],
        )
        oh.insert_after(model, conv, add)

    for i in range(len(contract.STRIDES)):
        cls_conv = oh.find_node(model, f"{head}/cv3.{i}/cv3.{i}.2/Conv")
        sig = oh.make_node(
            name=f"{head}/cv3.{i}/cv3.{i}.2/Sigmoid", op_type="Sigmoid",
            inputs=cls_conv.output, outputs=[f"class_prob_{i}"],
        )
        oh.insert_after(model, cls_conv, sig)


def _expose(model: onnx.ModelProto, node: onnx.NodeProto, out_name: str) -> None:
    """Re-point an existing producer's output to a canonical graph-output name
    (the reference `change_node_output`). Consumers of the old tensor are
    redirected so no dangling ref survives; the now-detached decode tail is
    dropped later by `save_model`'s `keep_reachable_from_outputs()`.
    """
    old = node.output[0]
    node.output[0] = out_name
    for n in model.graph.node:
        for k, inp in enumerate(n.input):
            if inp == old:
                n.input[k] = out_name


def _head_conv_depth(model: onnx.ModelProto, conv_name: str) -> int:
    """Out-channel count of a head Conv (from its weight initializer)."""
    conv = oh.find_node(model, conv_name)
    return int(oh.find_initializer_value(model, conv.input[1]).shape[0])


# --------------------------------------------------------------------------- #
# pose surgeon
# --------------------------------------------------------------------------- #
class SurgeonPose(SurgeonBase):
    name = "yolov8-pose"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        head, H, W = ident.head_prefix, ident.height, ident.width
        # keypoint head depth = 3 * num_kpts  (COCO pose: 51 = 3*17)
        kdepth = _head_conv_depth(model, f"{head}/cv4.0/cv4.0.2/Conv")
        log.info("pose surgery on %s: %d kpt channels (%d keypoints)",
                 head, kdepth, kdepth // 3)

        # 1) v8 detection base -> bbox_{i} + class_prob_{i}
        _rewrite_detection(model, ident)

        # 2) expose the 3 raw cv4 keypoint heads as kpt_{i} (no decode)
        for i, s in enumerate(contract.STRIDES):
            _expose(model, oh.find_node(model, f"{head}/cv4.{i}/cv4.{i}.2/Conv"), f"kpt_{i}")
            oh.add_output(model, f"kpt_{i}", (1, kdepth, H // s, W // s))

        return model


class SurgeonPoseV9(SurgeonPose):
    """v9-pose — GELAN backbone, identical v8 Detect+cv4 pose head; head-driven
    surgery applies verbatim. Registered for symmetry with the -seg family
    (Ultralytics ships no v9-pose checkpoint, but a custom v9-pose export
    identifies here)."""
    name = "yolov9-pose"


class SurgeonPoseV11(SurgeonPose):
    """v11-pose — structurally identical head at its own /model.N index; the
    head-driven surgery applies verbatim. Kept distinct so dispatch/metadata
    report `yolov11-pose`. (v12-pose routes through the v8-pose surgeon, as v12
    detect identifies as the v8 family.)"""
    name = "yolov11-pose"


class SurgeonYolo26Pose(SurgeonYolo26):
    """YOLO26 pose surgery. Unlike v8/v9/v11-pose (DFL detection base), yolo26-pose sits on the
    yolo26 **no-DFL / one2one** Detect base. The keypoint head is `{o}cv4.{i}` (3*num_kpts raw
    channels, `o=one2one_`). So surgery = the yolo26 box+class rewrite (`SurgeonYolo26`, which also
    lowers attention + prunes the E2E tail) PLUS re-exposing `{o}cv4.{i}` as `kpt_{i}` (no in-graph
    keypoint decode). Mirrors `SurgeonYolo26Seg` (kpt instead of mask, no proto). Contract: 9 tensors
    (bbox_/class_prob_/kpt_ x3)."""

    name = "yolo26-pose"

    def do_surgery(self, model: onnx.ModelProto, ident: YoloIdentity) -> onnx.ModelProto:
        head = ident.head_prefix
        o = ident.one2one_prefix or "one2one_"
        H, W = ident.height, ident.width
        kdepth = _head_conv_depth(model, f"{head}/{o}cv4_kpts.0/Conv")   # 3*num_kpts (yolo26 kpts conv)
        log.info("yolo26-pose surgery on %s: %d kpt channels (no-DFL/one2one)", head, kdepth)

        # 1) yolo26 no-DFL box + class base -> bbox_{i} + class_prob_{i} (clears outputs)
        super().do_surgery(model, ident)

        # 2) expose the 3 raw one2one_cv4 keypoint heads as kpt_{i}
        for i, s2 in enumerate(contract.STRIDES):
            _expose(model, oh.find_node(model, f"{head}/{o}cv4_kpts.{i}/Conv"), f"kpt_{i}")
            oh.add_output(model, f"kpt_{i}", (1, kdepth, H // s2, W // s2))
        return model
