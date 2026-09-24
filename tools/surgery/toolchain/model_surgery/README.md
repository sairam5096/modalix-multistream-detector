# model_surgery

Auto-identify a YOLO ONNX and rewrite its detection head so the graph exposes the
**SiMa generic box-decoder input contract** — the surgery front-end of the
Ultralytics -> SiMa conversion toolchain.

Self-contained by design: depends only on `onnx` + `numpy` (and, if present,
`onnxsim`/`onnxruntime`). No SiMa/Neat SDK and no Claude skills at runtime — it is
meant to run inside the Nx cloud and feed the AI Manager plugin.

## What it does

```
identify → audit → surgery → embed labels/metadata → save
        → re-audit → contract check → ORT numeric sanity → boxdecoder.json
```

1. **Identify** the model (version / flavor / H×W / num_classes / detection-head
   prefix) — no need to name the variant.
2. **Audit** operators against the vendored `data/supported_operators.json`
   (Modalix → `bfloat16`), before and after surgery.
3. **Surgery**: unroll the DFL and rewrite the head to emit, at strides 8/16/32:
   - `bbox_{0,1,2}` → `(1, 4, H/s, W/s)` — decoded, pixel-space `(cx,cy,w,h)`
   - `class_prob_{0,1,2}` → `(1, num_classes, H/s, W/s)` — post-Sigmoid
4. **Embed labels in the model** (`metadata_props`): a `labels` JSON array and the
   runtime string `formatted_labels = "bboxes-format:xyxysc;0:person;…"`, so the AI
   Manager plugin reads labels straight from the model file.
5. Validate: **contract check** (names/shapes) + **ORT numeric sanity**.
6. Emit `boxdecoder.json` sidecar for the pipeline builder.

## Usage

```bash
python -m model_surgery path/to/yolov8n.onnx --outdir ./out
# options:
#   --labels labels.txt|.json   (default: COCO80 if 80 classes, else class_<i>)
#   --variant yolov8            (force; overrides auto-ID)
#   --topk 25                   (box-decoder max detections)
#   --dtype bfloat16|int8|any   (audit policy; Modalix→bfloat16)
```

Outputs in `--outdir`: `<name>_surgery.onnx` (labels embedded) + `boxdecoder.json`.

## Variant coverage — all validated end-to-end on real exports

| Family | Contract | Head rewrite | Verified |
|--------|----------|--------------|----------|
| YOLOv8 | anchor-free 6-tensor | DFL unroll → cxcywh_pixel | box Δ 6e-5, cls 0 |
| YOLOv9 | anchor-free 6-tensor | = v8 (subclass) | box Δ 6e-5, cls 0 |
| YOLOv10 | anchor-free 6-tensor | one2one + DFL unroll, E2E-tail prune | box Δ 6e-5, cls 0 |
| YOLOv11 | anchor-free 6-tensor | = v8 (subclass) | box Δ 6e-5, cls 0 |
| YOLO26 | anchor-free 6-tensor | one2one, **no DFL**, E2E-tail prune | box Δ 6e-5, cls 0 |
| YOLOv6 | anchor-free 6-tensor | `detect.*`; auto DFL(rm=17)/no-DFL | box Δ 6e-5, cls 0 |
| YOLOX | **3-tensor** `4+1+nc` | re-expose decoupled `[reg,obj,cls]` → `raw_i` | Δ 0 |
| YOLOv5 | **3-tensor** `na*(5+nc)` | re-expose raw head convs → `raw_i` | Δ 0 |
| YOLOv7 | **3-tensor** `na*(5+nc)` | = v5 (subclass); `--variant yolov7` | Δ 0 (yolov7-tiny) |

Three box-decoder contracts (anchor-free / anchor-based / yolox); auto-identified,
no need to name the variant. The 3-tensor families are decoded in the **plugin**
(`../plugin_postproc/`), the anchor-free families decode in-graph. Add a family by
dropping a `SurgeonBase` subclass in `surgeons/` + a detector in `identify.py`.

## Layout

```
model_surgery/
├── cli.py            # orchestration (python -m model_surgery)
├── identify.py       # auto-identification
├── contract.py       # box-decoder contract + metadata/label embedding + validator
├── labels.py         # COCO80 + label loading + formatted-label string
├── audit.py          # in-repo op-support audit (vendored DB)
├── onnx_helpers.py   # self-contained ONNX graph-edit primitives
├── surgeons/         # base + registry + per-variant surgeons
└── data/supported_operators.json   # vendored SiMa op-support DB (release 2.1)
```
