# b1157_pack_tools: run SDK 2.1.x single-output model packs on the Modalix B1157 release

Scripts to run model packs compiled with ModelSDK 2.1.x on the B1157 Modalix platform release, whose runtime enforces a strict model-pack contract.
Python 3, numpy, `ml_dtypes`; `ev_transforms` (from the ModelSDK)
for the general de-tessellation path; `pyneat` on the SoM for the runner.

| Script | Purpose |
|---|---|
| `inspect_mpk.py PACK...` | Prints the plugin chain and a verdict: RUNS-AS-IS, NEEDS-REWRITE, NEEDS-RECOMPILE |
| `convert_mpk_b1157.py --in PACK --out PACK_b1157` | Rewrites a single-output pack (bare detessellate consumer) into the typed `unpack -> cast(float32) -> pass_through` chain the B1157 runtime accepts; keeps the tessellation geometry as `detess_geometry.json` inside the archive |
| `ofm_recover.py` | Library: `recover_words()` (bit-exact raw OFM words from the float32 output), `detessellate_from_geometry()` (general, via `ev_transforms`), `detessellate_single_channel()` (fast path for pooled outputs), `detessellate_rows()` (hidden-state style outputs). Also a self-check CLI on a saved `.npy` |
| `run_converted_model.py` | Minimal pyneat example: load the converted pack, run one image or tensor input, recover and de-tessellate, report latency |
| `example_siglip2/` | The complete SigLIP2 application on top of this (pack patcher, pipeline class, zero-shot app) |

## Workflow

```
python3 inspect_mpk.py model_mpk.tar.gz
# NEEDS-REWRITE ->
python3 convert_mpk_b1157.py --in model_mpk.tar.gz --out model_b1157_mpk.tar.gz --workdir /media/nvme/work
# on the SoM:
python3 run_converted_model.py --pack model_b1157_mpk.tar.gz --image sample.png --resize 256 256 --save out.npy
python3 ofm_recover.py --npy out.npy --pack model_b1157_mpk.tar.gz     # self-check of the de-tessellation
```

In your own application:

```python
import pyneat, numpy as np
from ofm_recover import recover_words, geometry_from_pack, detessellate_from_geometry

geom = geometry_from_pack("model_b1157_mpk.tar.gz")
out = model.run([tensor], timeout_ms=30000)[0].to_numpy()      # float32 [1, N]
logical = detessellate_from_geometry(recover_words(out), geom)  # [N, D, H, W, C] float32
```

## Facts the tools rely on (verified on B1157, 2026-09-24/25)

- The runtime derives the MLA output contract from the consumer plugin; a bare `detessellate` consumer
  is refused for 2-byte outputs (`no exact typed logical output contract`). Packs whose consumer is
  `unpack_transform` (all ModelSDK 2.1.x multi-output detectors) run unchanged.
- `unpack` alone or `unpack + pass_through` is rejected at plan admission (`no exact physical span`);
  the float32 cast is required. The cast maps each 16-bit word w to the float32 with bits `w << 16`.
- Recover the words with a shift, never with `astype(bfloat16)`: NaN-looking byte pairs get
  canonicalized by float casts and the data is silently corrupted (all-zero embeddings).
- The MLA stores 2-byte elements as byte planes inside each 16-channel block: `[16 low bytes][16 high
  bytes]`. `ev_transforms.detessellation(..., align_c16=True, cblock=True)` handles it; for a
  single-channel output the value of a block is `(block[16] << 8) | block[0]`.
- Tensor-input models take their logical input shape (e.g. `[64,1,768]` float32) through `Model.run`;
  the pack's own cast and tessellate stages run on the EV74. The inference-only `groups.mla` route with
  a pre-tessellated raw byte stream stalls at preflight on B1157.
- Limits: single-MLA, single-output packs only. Multi-output packs already carry the typed chain.
  SDK 2.0.0 packs with quantize/dequantize stages need a recompile. Split ("part1") packs load but
  their outputs are intermediate tensors.
- Failed or killed model loads leak CMA on this release; reboot before measurements.

Validated result: SigLIP2 base-patch32 (vision + text, bf16) zero-shot on the reference image
38.4 % / cos +0.1457 versus 40.7 % / +0.1466 on SDK 2.1.2; vision 9.9 ms, text 62 ms/caption.
