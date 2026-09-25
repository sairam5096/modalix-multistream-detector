"""Rewrite a legacy bf16 SiMa model pack for the B1157 strict-contract runtime.

The B1157 Neat runtime derives a typed logical contract for the MLA output from the consumer plugin.
Legacy packs feed MLA_0 into a CVU 'detessellate' stage that carries no typed shape, and for bf16
the runtime then mis-sizes the contract (logical = bytes x 2) and refuses the pack. Packs compiled
for B1157 feed MLA_0 into an EV74 'unpack_transform' with explicit bf16 tensor shapes instead.

This script replaces the detessellate consumer with: unpack_transform (one bf16 tensor whose byte
size equals the MLA OFM) -> cast_transform to float32 -> pass_through, mirroring a working pack, and
removes the detessellate stage from pipeline_sequence.json. The model output becomes the RAW
(still tessellated) OFM as float32; de-tessellation is done by the application in numpy.

Usage: patch_packs.py <src.tar.gz> <dst.tar.gz>
"""
import json, os, sys, tarfile, shutil, glob

src, dst = sys.argv[1], sys.argv[2]
NO_CAST = "--no-cast" in sys.argv[3:]
SHAPE = None
for a in sys.argv[3:]:
    if a.startswith("--shape="):
        SHAPE = [int(x) for x in a[len("--shape="):].split(",")]
work = "/tmp/mpkx/patch_" + os.path.basename(dst).replace(".tar.gz", "")
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
with tarfile.open(src) as t:
    t.extractall(work)
root = work
J = glob.glob(f"{root}/*_mpk.json")[0]; d = json.load(open(J))
pl = d["plugins"]
mla = [p for p in pl if p["processor"] == "MLA"][0]
ofm_bytes = mla["output_nodes"][0]["size"]; elems = ofm_bytes // 2
keep = [p for p in pl if p["sequence"] <= mla["sequence"]]
seq = mla["sequence"]
unpack = {"name": "MLA_0_ofm_unpack_transform", "sequence": seq + 1, "processor": "EV74",
          "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "unpack_transform",
                            "params": {"tensor_types": ["bfloat16"], "tensor_shapes": [SHAPE or [1, elems]],
                                       "input_shapes": [[1, ofm_bytes]], "output_shapes": [SHAPE or [1, elems]]}},
          "input_nodes": [{"name": "MLA_0", "size": ofm_bytes}],
          "output_nodes": [{"name": "MLA_0_ofm_unpack_transform_0", "type": "buffer", "size": ofm_bytes}],
          "type": "sgpProcess", "resources": {"executable": "kernel_name_tbd"}}
cast = {"name": "cast_1", "sequence": seq + 2, "processor": "EV74",
        "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "cast_transform",
                          "params": {"out_dtype": "float32", "input_shapes": [SHAPE or [1, elems]], "output_shapes": [SHAPE or [1, elems]]}},
        "input_nodes": [{"name": "MLA_0_ofm_unpack_transform_0", "size": ofm_bytes}],
        "output_nodes": [{"name": "cast_1/raw_ofm", "type": "buffer", "size": ofm_bytes * 2}],
        "type": "sgpProcess", "resources": {"executable": "kernel_name_tbd"}}
pt = {"name": "PassThrough", "sequence": seq + 3, "processor": "EV74",
      "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "pass_through", "params": {}},
      "input_nodes": [{"name": "cast_1/raw_ofm", "size": ofm_bytes * 2}],
      "output_nodes": [{"name": "pass_through_out_0", "type": "buffer", "size": ofm_bytes * 2}],
      "type": "sgpProcess", "resources": None}
if NO_CAST:
    pt["sequence"] = seq + 2
    pt["input_nodes"] = [{"name": "MLA_0_ofm_unpack_transform_0", "size": ofm_bytes}]
    pt["output_nodes"] = [{"name": "pass_through_out_0", "type": "buffer", "size": ofm_bytes}]
    d["plugins"] = keep + [unpack, pt]
else:
    d["plugins"] = keep + [unpack, cast, pt]
json.dump(d, open(J, "w"), indent=2)
P = f"{root}/pipeline_sequence.json"; s = json.load(open(P))
for p in s["pipelines"]:
    p["sequence"] = [x for x in p["sequence"] if x["kernel"] != "detessellate"]
json.dump(s, open(P, "w"), indent=2)
if os.path.exists(f"{root}/0_detessellate.json"):
    os.remove(f"{root}/0_detessellate.json")
with tarfile.open(dst, "w:gz") as t:
    for f in sorted(os.listdir(root)):
        t.add(os.path.join(root, f), arcname=f)
print(dst, "OFM bytes", ofm_bytes, "-> unpack bf16 [1,%d] -> cast f32 -> pass_through" % elems)
for p in d["plugins"]:
    print("  ", p["sequence"], p["name"], p["processor"], p.get("config_params", {}).get("kernel"))
