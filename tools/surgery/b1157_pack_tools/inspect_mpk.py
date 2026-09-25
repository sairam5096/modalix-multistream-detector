#!/usr/bin/env python3
"""Inspect a SiMa model pack (*.tar.gz) and say whether the B1157 runtime will accept it.

Usage: inspect_mpk.py MODEL_mpk.tar.gz [...]

Verdicts:
  RUNS-AS-IS      post-MLA chain starts with unpack_transform (typed shapes): load it unchanged
  NEEDS-REWRITE   single-output pack whose MLA output feeds a bare detessellate stage (bf16/int16 OFM):
                  use convert_mpk_b1157.py, then de-tessellate in the application (ofm_recover.py)
  NEEDS-RECOMPILE SDK 2.0.0 quantize/dequantize chain or multi-output detessellate route: re-export
                  with the ModelSDK that matches the B1157 platform release
"""
import glob, json, os, shutil, sys, tarfile, tempfile


def load_pack(path):
    tmp = tempfile.mkdtemp(prefix="mpk_")
    with tarfile.open(path) as t:
        t.extractall(tmp)
    j = glob.glob(os.path.join(tmp, "**", "*_mpk.json"), recursive=True)
    if not j:
        raise SystemExit(f"{path}: no *_mpk.json manifest found")
    root = os.path.dirname(j[0])
    return root, json.load(open(j[0]))


def summarize(path):
    root, d = load_pack(path)
    try:
        _summarize(path, root, d)
    finally:
        shutil.rmtree(os.path.dirname(root) if os.path.basename(os.path.dirname(root)).startswith("mpk_") else root, ignore_errors=True)


def _summarize(path, root, d):
    plugins = sorted(d["plugins"], key=lambda p: p["sequence"])
    mla = [p for p in plugins if p["processor"] == "MLA"]
    print(f"== {path}")
    print(f"   model_sdk_version: {d.get('model_sdk_version')}   inputs: {[(n['name'], n['size']) for n in d.get('input_nodes', [])]}")
    for p in plugins:
        k = p.get("config_params", {}).get("kernel") or p["processor"]
        outs = ", ".join(f"{n['name']}:{n['size']}" for n in p["output_nodes"])
        print(f"   {p['sequence']:>2} {p['processor']:<5} {k:<26} -> {outs}")
    pm = os.path.join(root, "0_process_mla.json")
    dtype = None
    if os.path.exists(pm):
        dtype = json.load(open(pm)).get("data_type")
        print(f"   MLA output data_type: {dtype}")
    if len(mla) != 1:
        print(f"   VERDICT: NEEDS-RECOMPILE ({len(mla)} MLA stages; only single-MLA packs handled here)")
        return
    after = [p for p in plugins if p["sequence"] > mla[0]["sequence"]]
    first = after[0].get("config_params", {}).get("kernel") if after else None
    kernels = [p.get("config_params", {}).get("kernel") for p in plugins]
    n_out = len(mla[0]["output_nodes"])
    if first == "unpack_transform":
        print("   VERDICT: RUNS-AS-IS (typed unpack chain)")
    elif "quantization_transform" in kernels or "dequantization_transform" in kernels:
        print("   VERDICT: NEEDS-RECOMPILE (SDK 2.0.0 quantize/dequantize chain; box-decode route facts missing on B1157)")
    elif first == "detessellation_transform" and n_out == 1:
        print("   VERDICT: NEEDS-REWRITE (single-output, bare detessellate consumer) -> convert_mpk_b1157.py")
    else:
        print(f"   VERDICT: NEEDS-RECOMPILE (post-MLA chain starts with {first}, {n_out} outputs)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for a in sys.argv[1:]:
        summarize(a)
