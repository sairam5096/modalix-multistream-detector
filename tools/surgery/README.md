# Automated surgery → compiled pack

**You provide only the model. This produces the deployable SiMa Modalix pack.**

`identify_model.py` (in [`../../models/`](../../models/)) *reads* a pack; it does
**not** create one. This directory is the other half: an automated
`ONNX → graph surgery → quantize + compile → pack` pipeline, wrapped so the whole
thing is one command — and the conversion scripts are **vendored right here** in
[`toolchain/`](toolchain/), so you run them **inside the SiMa `sima-neat`
container** with no separate package to install.

See [`../../docs/MODEL_SURGERY.md`](../../docs/MODEL_SURGERY.md) for *why* surgery
matters (it's the ~2.4× throughput lever for YOLOv6). This file is *how* to run it.

## What it does

```
your_model.onnx (or .pt)
   │
   ▼  1. graph surgery        splits the fused decode tail into raw per-stride
   │      (toolchain/model_surgery)  heads bbox_{0,1,2} + class_prob_{0,1,2}, writes
   │                            boxdecoder.json (family, num_classes, labels…)
   ▼  2. quantize + compile   ModelSDK PTQ (int8, real-image calibration) or
   │      (toolchain/quantize_compile.py)  bf16 (calibration-free); whole graph on the MLA
   ▼  3. emit                 <name>_surgery_mpk.tar.gz  (labels bundled)
   ▼  4. verify               identify_model.py confirms family/classes + writes
          (models/identify_model.py)  a matching labels.txt
```

Works for **YOLOv6, v8, v9, v10, v11, yolo26, yolox** (family auto-detected) at
**any class count** — surgery preserves the class dimension, so a 3-class,
80-class or custom model all convert the same way.

## Prerequisite: the `sima-neat` (Model Compiler) container

The compile step uses the SiMa **ModelSDK** (`afe`). Run everything **inside the
`sima-neat` / Model Compiler container**, which provides it — that's the only
requirement. (The surgery step alone needs just `onnx`/`numpy`.) The scripts are
vendored in [`toolchain/`](toolchain/); nothing else to download.

## Run it (one command, inside `sima-neat`)

```bash
# int8 (recommended for throughput) — give it your own domain calibration images:
./surger.sh /path/to/your_yolov6.onnx --calib /path/to/calib_images --name my-v6

# bf16 (no calibration set needed, ~lossless, larger/slower than int8):
./surger.sh /path/to/your_yolov6.onnx --bf16 --name my-v6
```

The wrapper activates the Model Compiler env if `activate-model-compiler` is on
PATH, checks that `afe` imports, then runs the vendored `convert_int8.py` /
`convert.py`. If `afe` isn't importable it tells you to run inside `sima-neat`.

You can also call the vendored scripts directly:

```bash
cd toolchain
PYTHONPATH=. python3 convert_int8.py your_yolov6.onnx out/ --calib-dir imgs/ --num-calib 50
# surgery only (onnx/numpy, no ModelSDK):
PYTHONPATH=. python3 -m model_surgery your_yolov6.onnx --outdir out/
```

Prefer the prebuilt image? `./surger.sh model.onnx --calib imgs/ --docker`
(needs the `sima-ultralytics-toolchain` image; `TOOLCHAIN_IMAGE` overrides).

## Output

```
surger_out/
  <name>_surgery_mpk.tar.gz   ← deploy this
  boxdecoder.json             ← decode_type, num_classes, strides, thresholds, labels
  labels.txt                  ← class names, one per line (count == num_classes)
  logs.json                   ← per-stage provenance (surgery / compile / emit)
```

`surger.sh` finishes by running `identify_model.py` on the pack, so a successful
run also prints the ready-to-paste `models:` config line. If surgery didn't take,
`identify_model.py` fails the run and points you at `logs.json`.

Options: `--int8`/`--bf16`, `--calib DIR`, `--num-calib N` (default 50),
`--imgsz N` (default 640), `--out DIR`, `--name NAME`, `--docker`.
`./surger.sh --help`.

## Deploy in this app

Drop the pack under [`../../models/`](../../models/) and add a `models:` entry
(the line `surger.sh` printed):

```yaml
models:
  - {name: my-v6, path: ../models/my-v6_surgery_mpk.tar.gz, decode_type: yolov6, labels: my_labels.txt, min_score: 0.30}
```

Copy the emitted `labels.txt` next to the pack. The on-device decoder takes
`num_classes` from the labels file, so its line count **must** equal the pack's
class count — which is why `surger.sh` emits a matching one for you.

## Calibration notes (int8)

- Use **real images from your deployment** for `--calib` — a few dozen
  representative frames. Good calibration data is what keeps int8 accurate.
- Surgery itself is numerically lossless (it only relocates the decode math);
  any int8 accuracy delta comes from quantization, not surgery, and would apply
  to a non-surgered model too. Verify accuracy host-side (decode the fp32 export
  and the int8-simulated pack on the same images) — never benchmark accuracy by
  eye on a board.

## Contents

- [`toolchain/`](toolchain/) — vendored SiMa conversion scripts (see its `NOTICE`):
  `model_surgery/` (surgery), `convert.py` / `convert_int8.py` (orchestrators),
  `quantize_compile.py` (ModelSDK compile).
- `surger.sh` — the one-command wrapper.

## Packs compiled with ModelSDK 2.1.x on the B1157 platform release

The B1157 runtime enforces a strict model-pack contract and refuses some 2.1.x packs
(single-output models whose MLA output feeds a bare `detessellate` stage; SDK 2.0.0
quantize/dequantize packs). Multi-output detector packs produced by this surgery flow carry the
typed `unpack_transform` chain and run unchanged. For the rest, see
[`b1157_pack_tools/`](b1157_pack_tools/): `inspect_mpk.py` gives a per-pack verdict,
`convert_mpk_b1157.py` rewrites a single-output pack without recompiling, and `ofm_recover.py`
recovers and de-tessellates the raw MLA output in the application.
