#!/usr/bin/env python3
"""Rewrite a single-output SDK 2.1.x model pack so the B1157 runtime accepts it (no recompilation).

The B1157 runtime derives the MLA output contract from the plugin that consumes MLA_0. Packs whose
consumer is a bare CVU 'detessellate' stage are refused for 2-byte outputs. This tool replaces that
consumer with the chain the B1157 compiler emits:

    MLA_0 -> unpack_transform (bfloat16, one tensor of exactly the OFM size)
          -> cast_transform (float32) -> pass_through

All three stages are required (unpack alone, or unpack + pass_through, is rejected by plan
admission). The detessellate stage is removed from pipeline_sequence.json and 0_detessellate.json
is kept in the archive under the name detess_geometry.json so the application can de-tessellate
the raw output (see ofm_recover.py). The MLA .elf and every other file are untouched.

Usage:
  convert_mpk_b1157.py --in MODEL_mpk.tar.gz --out MODEL_b1157_mpk.tar.gz [--dtype bfloat16]

After conversion the model returns the RAW tessellated MLA output as float32 (bit pattern
uint16 << 16). Recover it with ofm_recover.recover_words() and de-tessellate with
ofm_recover.detessellate_from_geometry().
"""
import argparse, glob, json, os, shutil, sys, tarfile, tempfile

ELEM = {"bfloat16": 2, "int16": 2, "float16": 2, "int8": 1, "uint8": 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(ELEM), help="element type of the MLA output (default bfloat16)")
    ap.add_argument("--workdir", default=None, help="scratch directory (default: a temp dir; avoid /tmp on the SoM, it is RAM-backed)")
    a = ap.parse_args()

    work = a.workdir or tempfile.mkdtemp(prefix="mpk_b1157_")
    shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
    with tarfile.open(a.src) as t:
        t.extractall(work)
    manifests = glob.glob(os.path.join(work, "**", "*_mpk.json"), recursive=True)
    if len(manifests) != 1:
        sys.exit(f"expected one *_mpk.json in the pack, found {manifests}")
    J = manifests[0]; root = os.path.dirname(J)
    d = json.load(open(J))
    plugins = sorted(d["plugins"], key=lambda p: p["sequence"])
    mla = [p for p in plugins if p["processor"] == "MLA"]
    if len(mla) != 1:
        sys.exit(f"pack has {len(mla)} MLA stages; this tool handles single-MLA packs only")
    mla = mla[0]
    if len(mla["output_nodes"]) != 1:
        sys.exit("pack has several MLA outputs; multi-output packs already use unpack_transform or need a recompile")
    after = [p for p in plugins if p["sequence"] > mla["sequence"]]
    first = after[0].get("config_params", {}).get("kernel") if after else None
    if first == "unpack_transform":
        sys.exit("pack already has a typed unpack chain: it runs on B1157 as is, nothing to convert")
    if first != "detessellation_transform":
        sys.exit(f"post-MLA chain starts with {first!r}; this tool only handles the bare detessellate consumer")

    ofm_bytes = mla["output_nodes"][0]["size"]
    esz = ELEM[a.dtype]
    if ofm_bytes % esz:
        sys.exit(f"OFM size {ofm_bytes} is not a multiple of {a.dtype} element size")
    elems = ofm_bytes // esz
    seq = mla["sequence"]
    keep = [p for p in plugins if p["sequence"] <= seq]
    unpack = {"name": "MLA_0_ofm_unpack_transform", "sequence": seq + 1, "processor": "EV74",
              "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "unpack_transform",
                                "params": {"tensor_types": [a.dtype], "tensor_shapes": [[1, elems]],
                                           "input_shapes": [[1, ofm_bytes]], "output_shapes": [[1, elems]]}},
              "input_nodes": [{"name": mla["output_nodes"][0]["name"], "size": ofm_bytes}],
              "output_nodes": [{"name": "MLA_0_ofm_unpack_transform_0", "type": "buffer", "size": ofm_bytes}],
              "type": "sgpProcess", "resources": {"executable": "kernel_name_tbd"}}
    cast = {"name": "cast_1", "sequence": seq + 2, "processor": "EV74",
            "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "cast_transform",
                              "params": {"out_dtype": "float32", "input_shapes": [[1, elems]], "output_shapes": [[1, elems]]}},
            "input_nodes": [{"name": "MLA_0_ofm_unpack_transform_0", "size": ofm_bytes}],
            "output_nodes": [{"name": "cast_1/raw_ofm", "type": "buffer", "size": elems * 4}],
            "type": "sgpProcess", "resources": {"executable": "kernel_name_tbd"}}
    pt = {"name": "PassThrough", "sequence": seq + 3, "processor": "EV74",
          "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "pass_through", "params": {}},
          "input_nodes": [{"name": "cast_1/raw_ofm", "size": elems * 4}],
          "output_nodes": [{"name": "pass_through_out_0", "type": "buffer", "size": elems * 4}],
          "type": "sgpProcess", "resources": None}
    d["plugins"] = keep + [unpack, cast, pt]
    json.dump(d, open(J, "w"), indent=2)

    P = os.path.join(root, "pipeline_sequence.json")
    if os.path.exists(P):
        s = json.load(open(P))
        for p in s.get("pipelines", []):
            p["sequence"] = [x for x in p["sequence"] if x.get("kernel") != "detessellate"]
        json.dump(s, open(P, "w"), indent=2)
    for g in glob.glob(os.path.join(root, "*_detessellate.json")):
        os.replace(g, os.path.join(root, "detess_geometry.json"))   # keep geometry for the app, hide from the runtime

    with tarfile.open(a.dst, "w:gz") as t:
        for f in sorted(os.listdir(root)):
            t.add(os.path.join(root, f), arcname=f)
    print(f"{a.dst}: OFM {ofm_bytes} bytes -> unpack {a.dtype}[1,{elems}] -> cast float32 -> pass_through")
    for p in d["plugins"]:
        print("  ", p["sequence"], p["name"], p["processor"], p.get("config_params", {}).get("kernel"))
    if not a.workdir:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
