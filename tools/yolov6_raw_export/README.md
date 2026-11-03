# yolov6_raw_export — YOLOv6 packs for Neat 0.6.0 and later

Neat 0.6.0 moved box decoding to the Kernels backend, where `decode_type: yolov6`
means **raw heads**: per-level l/t/r/b distances (4 channels, grid units) plus class
**logits**. A pack whose graph already decodes the boxes (the SiMa `model_surgery`
output in `tools/surgery`, `bbox_format: cxcywh_pixel`) is then mis-read as
distances, and every box becomes `(0,0)-(frame)`. Classes and scores stay correct,
only the geometry breaks. No runtime option recovers it
([sima-neat/core#1272](https://github.com/sima-neat/core/issues/1272)).

This script produces the pack the 0.6.0 decoder expects, from the ONNX you already
have (official YOLOv6 export, or the surgered one):

```bash
python3 tools/yolov6_raw_export/yolov6_raw_export.py my_yolov6.onnx my_yolov6_raw.onnx
```

It keeps the backbone/neck/heads and replaces the tail:

| head | source layer | output | content |
|---|---|---|---|
| `bbox_0..2` | `/detect/reg_preds.N/Conv` | `[1,4,H,W]` | raw l/t/r/b (grid units) |
| `cls_0..2` | `/detect/cls_preds.N/Conv` | `[1,nc,H,W]` | class logits (no sigmoid) |

For DFL variants (`reg_max > 0`, 68-channel reg heads) it keeps the projection in the
graph so the output is still 4-channel: `[1,4,HW,bins]` → Softmax(axis=3) → MatMul with
weights `0..reg_max` (default), or `--dfl exp` for a softmax-free Exp → grouped 1×1 Convs → Div.
Both compile fully on the MLA; a Transpose→Softmax(axis=1) form does not (ModelSDK accepts
Softmax only on its channel axis). Verified on a 3-class `reg_max=16` export: 1e-6 vs the
reference math in onnxruntime, and ±1 px vs a host reference decode when run on the DevKit.

Then compile as usual (ModelSDK 3.0.0, int8 or bf16) and run with **all three** options:

```yaml
models:
  - {name: my-v6, path: ../models/my_yolov6_raw_mpk.tar.gz,
     decode_type: yolov6, decode_type_option: GroupedByRoleLogit, num_classes: 80,
     labels: my_labels.txt, min_score: 0.30}
```

`decode_type_option: GroupedByRoleLogit` is mandatory for this output order (3 bbox heads,
then 3 class heads). With the default `Auto` the runtime pairs the heads per level and the
pack fails to build (`BoxDecode kernel recipe: logical channel count exceeds source storage`).

## Class count (`num_classes`)

The script does not assume a class count. It cuts the graph at the model's own `cls_preds`
convolutions, so the `cls_N` outputs have whatever depth the model was trained with (80 for
COCO, 3 for a custom 3-class model, and so on). The last lines it prints tell you the value:

```
Detected num_classes = 3. Run the compiled pack with:
  decode_type=yolov6  decode_type_option=GroupedByRoleLogit  num_classes=3
```

The compiled pack does **not** carry a class count that the 0.6.0 decoder trusts, so the
runtime cannot detect it. You must pass `num_classes` yourself (config `num_classes:`,
PyNeat `ModelOptions.num_classes`, C++ `Model::Options::num_classes`) and it must equal the
printed value. Leaving it out fails the graph build with
`BoxDecode kernel recipe: class count must be 1..336`; a wrong value shifts or drops class
ids silently. Keep your labels file in the same order and length as the model's classes.

Do **not** run `tools/surgery` on the result: that is the step that folds the decode back in.
Use `tools/surgery` for Neat 0.3–0.5 runtimes, this script for 0.6.0 and later.

Dependencies: `onnx` (and `onnxruntime` only if you want to diff against the source model).
