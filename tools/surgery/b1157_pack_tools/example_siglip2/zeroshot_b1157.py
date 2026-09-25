"""Zero-shot SigLIP2 on Modalix with the B1157-patched packs (see patch_packs.py / pipeline_b1157.py).

Example:
  python apps/zeroshot_b1157.py --base_dir $PWD --image cats.png \
      --texts "two cats on a couch" "a dog running" "a red car" [--bench 200] [--vision_detess c16]
Expected on cats.png (from DELIVERY_NOTES): "two cats on a couch" ~40.7 %, cos ~+0.147, others ~0 %.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from siglip_modalix.pipeline_b1157 import SiglipModalixB1157
from zeroshot import bench


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--texts", nargs="+", required=True)
    ap.add_argument("--base_dir", default="/media/nvme/siglip_e2e")
    ap.add_argument("--tess", type=int, nargs=2, default=(16, 48))
    ap.add_argument("--detess", type=int, nargs=2, default=(3, 192))
    ap.add_argument("--vision_detess", default="c16", choices=["c16", "planes", "first", "ref"])
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--duration", type=float, default=0, help="keep running zero-shot for N seconds, report every 10 s")
    a = ap.parse_args()

    t0 = time.time()
    m = SiglipModalixB1157(a.base_dir, tess=tuple(a.tess), detess=tuple(a.detess), vision_detess=a.vision_detess)
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    probs, cos = m.zeroshot(a.image, a.texts)
    print(f"\nimage: {a.image}  (vision_detess={a.vision_detess})")
    for j in np.argsort(-probs):
        print(f"  {100*probs[j]:6.2f}%   (cos {cos[j]:+.4f})   {a.texts[j]}")
    sys.stdout.flush()

    if a.bench > 0:
        bench(m, a.image, a.texts, a.bench)
    if a.duration > 0:
        from PIL import Image
        img = np.array(Image.open(a.image).convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8)
        end = time.time() + a.duration; nxt = time.time() + 10; lat = []; n = 0; err = 0
        while time.time() < end:
            s = time.perf_counter()
            try:
                m.zeroshot(img, a.texts); lat.append((time.perf_counter() - s) * 1e3); n += 1
            except Exception as ex:
                err += 1; print(f"ERROR {ex}", flush=True); time.sleep(0.5)
            if time.time() >= nxt:
                print(f"[run] t={int(a.duration-(end-time.time())):4d}s n={n} err={err} e2e median {np.median(lat):.1f} ms p95 {np.percentile(lat,95):.1f} ms", flush=True)
                lat = []; nxt += 10
        print(f"[done] {n} zero-shot queries, {err} errors", flush=True)


if __name__ == "__main__":
    main()
