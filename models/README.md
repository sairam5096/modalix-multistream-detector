# Models

Model packs are **not** included in this repo (licensing + size). The configs
reference compiled SiMa Modalix packs (`.tar.gz`) by relative path:

| Config name | Pack | Precision | Role |
|---|---|---|---|
| `coco-n`  | `yolo26n-det-int8-b1.tar.gz`         | INT8 | fast, high throughput |
| `coco-v8` | `yolov8n_ultra_mpk.tar.gz`           | INT8 | alternate |
| `coco-m`  | `yolo26m-det-bf16-mla_tess-b1.tar.gz`| BF16 | accurate |

Compile your own with the SiMa ModelSDK / `sima-cli` quantize-compile flow
(surgery → quantize → compile for `modalix`), then drop the `.tar.gz` files
here and point the config `path:` fields at them. A `coco_label.txt` (80 COCO
class names, one per line) is also expected next to the detector.

> All models decode as `decode_type: yolo26` (direct box regression) — including
> the YOLOv8 surgery pack, which folds DFL into direct 4-channel heads.
