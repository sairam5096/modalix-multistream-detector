# YOLOv6 model surgery — the biggest throughput lever

A stock YOLOv6 ONNX export ends in a **fused decode tail**: a single output
tensor `[1, 8400, 85]` produced by concatenating the three detection heads and
running the box-decode arithmetic (DFL / grid + stride, objectness × class)
*inside the graph*. On the SiMa MLA that tail does not map to the accelerator —
it **falls back to the A65 CPU**, and that fallback caps each inference pipeline
far below the MLA's real rate.

**Surgery** rewrites the graph so the decode tail is removed and the three heads
are exposed as **six raw per-stride tensors** that the on-device
`neatobjectdecode` plugin consumes directly. The whole graph then runs on the
MLA with no A65 fallback.

```
stock export            outputs: [1, 8400, 85]            (decode on A65)
        │  surgery
        ▼
surgered export         bbox_0       [1,  4, 80, 80]      \
                        bbox_1       [1,  4, 40, 40]       |  4-channel box reg
                        bbox_2       [1,  4, 20, 20]       /  per stride 8/16/32
                        class_prob_0 [1, 80, 80, 80]      \
                        class_prob_1 [1, 80, 40, 40]       |  N-class probs
                        class_prob_2 [1, 80, 20, 20]      /  per stride 8/16/32
                        (decode moves to the neatobjectdecode plugin)
```

`N` is the class count — surgery preserves it, so a 3-class, 80-class or any
custom YOLOv6 surgers the same way. The `class_prob_*` heads carry the raw class
logits (`class_is_prob: false`; the on-device decoder applies the sigmoid — see the
score-semantics section below), and `bbox_*` are the raw reg heads the decoder
turns into `cxcywh_pixel` → `xyxysc` boxes.

## Why it matters (measured)

On one Modalix (PCIe card, SDK 2.1.3), single serial pipeline, 720p sources:

| YOLOv6s pack | single-pipeline ceiling | notes |
|---|---|---|
| **stock (decode on A65)** | **~30 fps** | the tail serializes on the CPU |
| **surgered (whole graph on MLA)** | **~119 fps** | ~**2.4×** faster |

This is a larger lever than any runtime knob (`max_inflight_*`, pipeline count,
etc.). With a surgered model, a single shared pipeline keeps up with far more
streams before you ever need to split work across multiple pipelines. See
[BENCHMARKS.md](BENCHMARKS.md) for the full sweep.

> Accuracy is preserved. Surgery is a *graph-equivalent* transform — it moves
> the decode math out of the graph but does not change the underlying
> convolutional features, so the detections match the stock model. (Verify
> host-side by decoding both exports on the same images and IoU-matching the
> boxes; do not benchmark accuracy on-device.)

## The recipe

You need the SiMa **ModelSDK** (or `sima-cli` quantize-compile flow) and the
model-surgery tooling for your SDK version. Inputs: your trained YOLOv6 ONNX and
a folder of real calibration images (a few dozen frames representative of your
deployment).

1. **Export** your YOLOv6 to ONNX with the detection heads intact (a "no-NMS"
   export is fine — surgery replaces the decode tail regardless).
2. **Surgery** — run the YOLOv6 surgeon to split the fused tail into the six
   `bbox_*` / `class_prob_*` heads above. The surgeon writes a `boxdecoder.json`
   describing the decode contract (`decode_type: yolov6`, `strides`,
   `input_depth [4,4,4,N,N,N]`, `num_classes`, and the embedded labels).
3. **Quantize + compile** the surgered ONNX for `modalix` with real-image
   calibration (INT8 recommended for throughput; BF16 if you need it). This
   produces the deployable `*_mpk.tar.gz` pack.
4. **Verify the pack** with [`../models/identify_model.py`](../models/identify_model.py):

   ```bash
   python3 models/identify_model.py path/to/your_surgery_mpk.tar.gz --name my-v6
   # Identified: yolov6 (decode_type=yolov6, v6)
   #   classes    : N
   #   input      : 640x640  strides=[8, 16, 32]  head_depth=[4,4,4,N,N,N]
   ```

   If `identify_model.py` reports the pack as **not directly usable**
   (`nonms` / no box-decode head), the surgery step did not take — re-run it
   before compiling.

## Using the surgered pack in this app

Point a `models:` entry at the pack with `decode_type: yolov6` and a labels file
whose line count equals the pack's `num_classes` (the decoder reads the class
count from the labels file — an 80-line `coco_label.txt` under a 3-class model
decodes wrong). `identify_model.py --write-labels` emits a matching labels file
straight from the pack:

```bash
python3 models/identify_model.py my_surgery_mpk.tar.gz --name my-v6 \
        --write-labels my_labels.txt
# then in the config:
#   - {name: my-v6, path: ../models/my_surgery_mpk.tar.gz, decode_type: yolov6, labels: my_labels.txt, min_score: 0.30}
```

## When you still need multiple pipelines

Surgery raises the single-pipeline ceiling; it does not make it infinite. If a
single surgered pipeline still cannot keep up with your stream count, *then*
split the load across multiple pipelines (declare the model more than once, or
run multiple containers). But note: **once a surgered pipeline is already near
the MLA ceiling, adding a second pipeline does not help** — both pipelines
contend for the same MLA and total throughput is flat-to-slightly-worse.
Multiple pipelines only add throughput while a single pipeline is *below* the
MLA ceiling. Surger first; split only if you must.

## Score semantics: the class tensors are logits (fixed 2026-10-06)

Earlier versions of the surgery emitted a `Sigmoid` on the class head and recorded
`class_is_prob: true`. The on-device Neat box decoder for `decode_type` **`yolov6`**,
**`yolox`** and **`yolo26`** applies its own sigmoid and cannot be told not to
(`class_is_prob` never reaches the runtime), so scores came out as
`sigmoid(sigmoid(x))`: never below 0.5, never above 0.731. A `min_score` of 0.45
filtered nothing, and 0.8 returned nothing. Boxes and ranking were unaffected, which
is why it passed visual checks. Found and analysed by Wobot; reproduced here on a
Modalix card (yolov6s int8: on-device scores 0.54–0.72 with `min_score: 0.01`).

Now the surgeons for those decode types emit the raw conv output (logits) and write
`class_is_prob: false`; the decoder's single sigmoid gives real probabilities, so
`min_score` means what it says. `--class-output logits|prob` forces either shape, and
surgery warns if a pack would double-sigmoid.

**Packs built before this change must be re-run through `surger.sh`** — the fix is in
the graph, not the runtime. Re-check your `min_score` values afterwards: on an old pack
every value at or below 0.5 behaved like 0.
