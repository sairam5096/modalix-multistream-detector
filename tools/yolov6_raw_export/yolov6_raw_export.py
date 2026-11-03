#!/usr/bin/env python3
"""Cut a YOLOv6 ONNX export back to raw heads for Neat 0.6.0 BoxDecodeType::YoloV6.

Neat 0.6.0 decodes `yolov6` as raw l/t/r/b (grid units) + class logits. This script removes the
in-graph dist2bbox / sigmoid tail of a YOLOv6 export (official export or SiMa-surgered) and emits
six outputs grouped by role: bbox_0..2 ([1,4,H,W], raw ltrb) then cls_0..2 ([1,nc,H,W], logits).

Usage: yolov6_raw_export.py in.onnx out.onnx
Then compile as usual and run with decode_type=yolov6, decode_type_option=GroupedByRoleLogit, num_classes=nc.
"""
import re, sys, onnx
from onnx import shape_inference

src, dst = sys.argv[1], sys.argv[2]
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
    # MLA-friendly DFL: bins on the channel axis -> Softmax(axis=1) -> 1x1 Conv with weights 0..nb-1
    # (this is YOLOv6's own proj_conv formulation).
    s0 = numpy_helper.from_array(np.array([1, 4, nb, H * W], np.int64), name + "_s0")
    s1 = numpy_helper.from_array(np.array([1, 4, H, W], np.int64), name + "_s1")
    proj = numpy_helper.from_array(np.arange(nb, dtype=np.float32).reshape(1, nb, 1, 1), name + "_proj")
    new_inits += [s0, s1, proj]
    new_nodes += [
        helper.make_node("Reshape", [name + "_dfl", s0.name], [name + "_r"]),            # [1,4,nb,HW]
        helper.make_node("Transpose", [name + "_r"], [name + "_t"], perm=[0, 2, 1, 3]),   # [1,nb,4,HW]
        helper.make_node("Softmax", [name + "_t"], [name + "_sm"], axis=1),
        helper.make_node("Conv", [name + "_sm", proj.name], [name + "_e"], kernel_shape=[1, 1]),  # [1,1,4,HW]
        helper.make_node("Reshape", [name + "_e", s1.name], [name]),                     # [1,4,H,W]
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
