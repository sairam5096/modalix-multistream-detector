# Automated surgery → compiled pack

**You provide only the model. This produces the deployable SiMa Modalix pack.**

`identify_model.py` (in [`../../models/`](../../models/)) *reads* a pack; it does
**not** create one. This directory is the other half: an automated
`ONNX → graph surgery → quantize + compile → pack` pipeline, wrapped so the whole
thing is one command.

See [`../../docs/MODEL_SURGERY.md`](../../docs/MODEL_SURGERY.md) for *why* surgery
matters (it's the ~2.4× throughput lever for YOLOv6). This file is *how* to run it.

## What it does

```
your_model.onnx (or .pt)
   │
   ▼  1. graph surgery        splits the fused decode tail into raw per-stride
   │      (model_surgery)       heads  bbox_{0,1,2} + class_prob_{0,1,2}, writes
   │                            boxdecoder.json (family, num_classes, labels…)
   ▼  2. quantize + compile   ModelSDK PTQ (int8, real-image calibration) or
   │      (ModelSDK, modalix)   bf16 (calibration-free), whole graph on the MLA
   ▼  3. emit                 <name>_surgery_mpk.tar.gz  (labels bundled)
   ▼  4. verify               identify_model.py confirms family/classes + writes
          (identify_model.py)   a matching labels.txt
```

Works for **YOLOv6, v8, v9, v10, v11, yolo26, yolox** (the family is
auto-detected), at **any class count** — surgery preserves the class dimension,
so a 3-class, 80-class or custom model all convert the same way.

## Prerequisite: the SiMa conversion toolchain image

Surgery + compile run inside the SiMa toolchain container
`sima-ultralytics-toolchain`, which bundles the ModelSDK compiler. Obtain or
build it from the **SiMa Modalix SDK** (it needs the ModelSDK, which is licensed
through SiMa — the same SDK you already use to compile models). Confirm it's
present:

```bash
docker images | grep sima-ultralytics-toolchain
```

Set `TOOLCHAIN_IMAGE=<name>` if yours is tagged differently.

## Run it (one command)

```bash
# int8 (recommended for throughput) — give it your own domain calibration images:
./surger.sh /path/to/your_yolov6.onnx --calib /path/to/calib_images --name my-v6

# bf16 (no calibration set needed, ~lossless, larger/slower than int8):
./surger.sh /path/to/your_yolov6.onnx --bf16 --name my-v6
```

Output (default `surger_out/` next to the model):

```
surger_out/
  <name>_surgery_mpk.tar.gz   ← deploy this
  boxdecoder.json             ← decode_type, num_classes, strides, thresholds, labels
  labels.txt                  ← class names, one per line (count == num_classes)
  logs.json                   ← per-stage provenance (surgery / compile / emit)
```

`surger.sh` finishes by running `identify_model.py` on the pack, so a successful
run *also* prints the ready-to-paste `models:` config line. If surgery didn't
take, `identify_model.py` fails the run and points you at `logs.json`.

Options: `--int8`/`--bf16`, `--calib DIR`, `--num-calib N` (default 50),
`--imgsz N` (default 640), `--out DIR`, `--name NAME`. `./surger.sh --help`.

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
