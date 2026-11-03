#!/usr/bin/env python3
"""Cut a YOLOv6 ONNX export back to raw heads for Neat 0.6.0 BoxDecodeType::YoloV6.

Neat 0.6.0 decodes `yolov6` as raw l/t/r/b (grid units) + class logits. This script removes the
in-graph dist2bbox / sigmoid tail of a YOLOv6 export (official export or SiMa-surgered) and emits
six outputs grouped by role: bbox_0..2 ([1,4,H,W], raw ltrb) then cls_0..2 ([1,nc,H,W], logits).

Usage: yolov6_raw_export.py in.onnx out.onnx [--dfl softmax|exp]
Then compile as usual and run with decode_type=yolov6, decode_type_option=GroupedByRoleLogit, num_classes=nc.
"""
import re, sys, onnx
from onnx import shape_inference

args = [a for a in sys.argv[1:] if not a.startswith("--")]
DFL_MODE = "exp" if "--dfl" in sys.argv and sys.argv[sys.argv.index("--dfl") + 1] == "exp" else "softmax"
if len(args) < 2:
    sys.exit(__doc__)
src, dst = args[0], args[1]
m = onnx.load(src)
g = m.graph
produced = {o: n for n in g.node for o in n.output}

def find(patterns):
    for pat in patterns:
        rx = re.compile(pat)
        hits = sorted({o for o in produced if rx.search(o)}, key=lambda s: int(re.search(r"\.(\d)/", s).group(1)))
        if len(hits) == 3:
            return hits
    return None

# raw ltrb: DFL variants expose proj_conv (4ch after softmax*proj); plain variants expose reg_preds (4ch)
bbox = find([r"/detect/proj_conv\.(\d)/.*Conv_output_0$", r"/detect/reg_preds\.(\d)/Conv_output_0$"])
cls = find([r"/detect/cls_preds\.(\d)/Conv_output_0$"])
if not bbox or not cls:
    sys.exit(f"could not find YOLOv6 head tensors (bbox={bbox}, cls={cls}); pass an export with original node names")
print("bbox heads:", bbox); print("cls heads :", cls)
onnx.utils.extract_model(src, dst, [g.input[0].name], bbox + cls)
r = onnx.load(dst)
names = [f"bbox_{i}" for i in range(3)] + [f"cls_{i}" for i in range(3)]
for o, nn in zip(r.graph.output, names):
    old = o.name
    for n in r.graph.node:
        for k, x in enumerate(n.output):
            if x == old: n.output[k] = nn
    o.name = nn
r = shape_inference.infer_shapes(r)

# DFL variants (reg_max>0): reg_preds depth is 4*(reg_max+1). Neat's yolov6 profile wants 4-ch raw ltrb,
# so keep the DFL projection in-graph (softmax over bins, expectation) and stop before dist2bbox.
import numpy as np
from onnx import helper, numpy_helper, TensorProto
shape_of = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in list(r.graph.value_info) + list(r.graph.output)}
new_nodes, new_inits = [], []
for i in range(3):
    name = f"bbox_{i}"
    _, C, H, W = shape_of[name]
    if C == 4:
        continue
    nb = C // 4
    assert nb * 4 == C, f"{name}: depth {C} is not 4*bins"
    print(f"{name}: DFL head with {nb} bins -> adding softmax projection")
    for n in r.graph.node:
        for k, x in enumerate(n.output):
            if x == name: n.output[k] = name + "_dfl"
    # MLA-friendly DFL. The ModelSDK only accepts Softmax on what it sees as the channel axis and
    # fails on a Transpose+Softmax(axis=1) form in the MLA simulator, so put the bins LAST:
    # [1,4,nb,HW] -> Transpose -> [1,4,HW,nb] -> Softmax(axis=3) -> MatMul [nb,1] -> [1,4,HW,1] -> [1,4,H,W].
    # (--dfl exp: softmax-free alternative, Exp + two grouped 1x1 convs + Div; also all-MLA.)
    s0 = numpy_helper.from_array(np.array([1, 4, nb, H * W], np.int64), name + "_s0")
    s1 = numpy_helper.from_array(np.array([1, 4, H, W], np.int64), name + "_s1")
    if DFL_MODE == "exp":
        wn = np.zeros((4, nb, 1, 1), np.float32); wd = np.ones((4, nb, 1, 1), np.float32)
        for a in range(4): wn[a, :, 0, 0] = np.arange(nb)
        Wn = numpy_helper.from_array(wn, name + "_wn"); Wd = numpy_helper.from_array(wd, name + "_wd")
        new_inits += [Wn, Wd]
        new_nodes += [
            helper.make_node("Exp", [name + "_dfl"], [name + "_ex"]),
            helper.make_node("Conv", [name + "_ex", Wn.name], [name + "_num"], kernel_shape=[1, 1], group=4),
            helper.make_node("Conv", [name + "_ex", Wd.name], [name + "_den"], kernel_shape=[1, 1], group=4),
            helper.make_node("Div", [name + "_num", name + "_den"], [name]),
        ]
    else:
        proj = numpy_helper.from_array(np.arange(nb, dtype=np.float32).reshape(nb, 1), name + "_proj")
        new_inits += [s0, s1, proj]
        new_nodes += [
            helper.make_node("Reshape", [name + "_dfl", s0.name], [name + "_r"]),
            helper.make_node("Transpose", [name + "_r"], [name + "_t"], perm=[0, 1, 3, 2]),
            helper.make_node("Softmax", [name + "_t"], [name + "_sm"], axis=3),
            helper.make_node("MatMul", [name + "_sm", proj.name], [name + "_e"]),
            helper.make_node("Reshape", [name + "_e", s1.name], [name]),
        ]
    for o in r.graph.output:
        if o.name == name:
            o.type.tensor_type.shape.dim[1].dim_value = 4
if new_nodes:
    r.graph.node.extend(new_nodes); r.graph.initializer.extend(new_inits)
    del r.graph.value_info[:]
    r = shape_inference.infer_shapes(r)
onnx.checker.check_model(r)
onnx.save(r, dst)
for o in r.graph.output:
    print(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
ncs = {o.type.tensor_type.shape.dim[1].dim_value for o in r.graph.output if o.name.startswith("cls_")}
assert len(ncs) == 1, f"class heads disagree on channel count: {ncs}"
nc = ncs.pop()
print(f"\nDetected num_classes = {nc}. Run the compiled pack with:")
print(f"  decode_type=yolov6  decode_type_option=GroupedByRoleLogit  num_classes={nc}")
