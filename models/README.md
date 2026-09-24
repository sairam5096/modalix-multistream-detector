# Models

Model packs are **not** included in this repo (licensing + size). The configs
reference compiled SiMa Modalix packs (`.tar.gz`) by relative path:

| Config name | Pack | Precision | Role |
|---|---|---|---|
| `coco-n`  | `yolo26n-det-int8-b1.tar.gz`         | INT8 | fast, high throughput |
| `coco-v8` | `yolov8n_ultra_mpk.tar.gz`           | INT8 | alternate |
| `coco-m`  | `yolo26m-det-bf16-mla_tess-b1.tar.gz`| BF16 | accurate |
| `coco-v6` | `yolov6n_surgery_mpk.tar.gz`         | INT8 | YOLOv6n — see `som/configs/example-yolov6-multimodel.yaml` |

Compile your own with the SiMa ModelSDK / `sima-cli` quantize-compile flow
(surgery → quantize → compile for `modalix`), then drop the `.tar.gz` files
here and point the config `path:` fields at them. A `coco_label.txt` (80 COCO
class names, one per line) is also expected next to the detector.

> Most packs decode as `decode_type: yolo26` (direct box regression) — including
> the YOLOv8 surgery pack, which folds DFL into direct 4-channel heads. The
> **YOLOv6** pack is the exception: it uses `decode_type: yolov6` (a 4-channel
> reg + N-class head at 640×640). Set `decode_type` per model in the config;
> the wrong decoder yields empty or malformed boxes.

## Identifying a pack (`identify_model.py`)

Don't guess the `decode_type`, class count, or labels — read them off the pack.
`identify_model.py` inspects a compiled `.tar.gz` (or unpacked dir) and prints the
detector family, the **number of classes**, the input geometry, and a ready-to-paste
`models:` entry. This is what makes a YOLOv6 model with *any* class count — 3-class,
80-class, custom — drop into the app correctly: the detector's `num_classes` follows
the labels file, so a mismatched labels file (e.g. an 80-class `coco_label.txt` under
a 3-class model) decodes wrong. The tool reads the real class count and can write a
matching labels file.

```bash
# identify + get the config line
python3 models/identify_model.py models/yolov6s_surgery_mpk.tar.gz --name coco-v6s
#   Identified: yolov6 (decode_type=yolov6, v6)
#     classes    : 80
#     input      : 640x640  strides=[8, 16, 32]  head_depth=[4,4,4,80,80,80]
#   - {name: coco-v6s, path: ../models/yolov6s_surgery_mpk.tar.gz, decode_type: yolov6, labels: labels.txt, min_score: 0.30}

# a custom 3-class YOLOv6 → write the matching labels file automatically
python3 models/identify_model.py models/ppe_yolov6n_mpk.tar.gz --name ppe \
        --write-labels ppe_labels.txt
#     classes    : 3   ->  wrote 3 labels -> ppe_labels.txt

python3 models/identify_model.py <pack> --json   # machine-readable
```

- **YOLOv6 surgery packs** carry a `boxdecoder.json` with the full identity
  (`model_family`, `num_classes`, `input_depth`, embedded labels) — fully read.
- **yolo26 / yolov8 "raw" packs** have no box-decode head baked in; family is taken
  from the pack name and `num_classes` from the output tensor shape, and they decode
  via `decode_type` + a labels file of that many entries.
- **`nonms` / un-surgered YOLOv6 packs** (no compatible box-decode head) are reported
  as not directly usable — re-compile them through the SiMa **model-surgery** flow
  (which folds the head into direct 4-channel reg + N-class outputs) so the on-device
  `decode_type: yolov6` decoder can consume them. Surgery also runs the whole graph on
  the MLA (no A65 fallback), which is typically a large throughput win.
