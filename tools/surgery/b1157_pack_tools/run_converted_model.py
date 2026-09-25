#!/usr/bin/env python3
"""Minimal pyneat example for a pack converted with convert_mpk_b1157.py (single MLA output).

Loads the pack, runs one input, recovers the raw OFM words and de-tessellates them with the
geometry stored in the pack. Works for image models (--image) and tensor models (--tensor_shape).

  python run_converted_model.py --pack vision_b1157_mpk.tar.gz --image cats.png --resize 256 256
  python run_converted_model.py --pack text_b1157_mpk.tar.gz --tensor_shape 64,1,768
"""
import argparse, time
import numpy as np
import pyneat
from ofm_recover import recover_words, geometry_from_pack, detessellate_from_geometry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--image", help="run as an image model with on-CVU preprocess")
    ap.add_argument("--resize", type=int, nargs=2, default=(256, 256))
    ap.add_argument("--mean", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    ap.add_argument("--std", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    ap.add_argument("--tensor_shape", help="run as a tensor model: logical input shape H,W,C (e.g. 64,1,768), random input")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--save", help="save the raw float32 runtime output as .npy")
    a = ap.parse_args()

    o = pyneat.ModelOptions(); pp = o.preprocess
    if a.image:
        from PIL import Image
        pp.kind = pyneat.InputKind.Image; pp.enable = pyneat.AutoFlag.On
        cc = pp.color_convert; cc.enable = pyneat.AutoFlag.On
        cc.input_format = pyneat.PreprocessColorFormat.RGB; cc.output_format = pyneat.PreprocessColorFormat.RGB; pp.color_convert = cc
        rs = pp.resize; rs.enable = pyneat.AutoFlag.On; rs.width, rs.height = a.resize; pp.resize = rs
        nm = pp.normalize; nm.enable = pyneat.AutoFlag.On; nm.mean = list(a.mean); nm.stddev = list(a.std); pp.normalize = nm
        img = np.array(Image.open(a.image).convert("RGB").resize(tuple(a.resize), Image.BILINEAR), dtype=np.uint8)
        def make():
            return pyneat.Tensor.from_numpy(np.ascontiguousarray(img), copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74)
    else:
        H, W, C = (int(x) for x in a.tensor_shape.split(","))
        pp.kind = pyneat.InputKind.Tensor
        pp.input_max_width, pp.input_max_height, pp.input_max_depth = W, H, C
        x = np.random.randn(H, W, C).astype(np.float32)
        def make():
            return pyneat.Tensor.from_numpy(np.ascontiguousarray(x), copy=True, memory=pyneat.TensorMemory.EV74)

    t0 = time.time(); m = pyneat.Model(a.pack, o); print(f"loaded in {time.time()-t0:.1f}s")
    out = m.run([make()], timeout_ms=30000)[0].to_numpy()
    print("runtime output", out.dtype, out.shape, out.nbytes, "bytes")
    if a.save:
        np.save(a.save, out)
    words = recover_words(out)
    logical = detessellate_from_geometry(words, geometry_from_pack(a.pack))
    print("logical output", logical.shape, "min %.4g max %.4g" % (np.nanmin(logical), np.nanmax(logical)))
    lat = []
    for _ in range(a.iters):
        s = time.perf_counter(); m.run([make()], timeout_ms=30000); lat.append((time.perf_counter() - s) * 1e3)
    print(f"median {np.median(lat):.2f} ms over {a.iters} runs")


if __name__ == "__main__":
    main()
