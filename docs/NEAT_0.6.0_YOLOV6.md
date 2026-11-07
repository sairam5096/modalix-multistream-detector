# YOLOv6 on Neat 0.6.0: boxes at (0,0), frame-sized — cause and fix

**Symptom.** A YOLOv6 pack that detects correctly on Neat 0.3–0.5 (and on the 0.6 EA builds)
returns, on Neat 0.6.0 / SDK 3.0.0, boxes that all start at `(0,0)` and mostly cover the whole
frame. Classes and scores are still right.

**Cause.** 0.6.0 moved box decoding into the Kernels backend. There `decode_type: yolov6`
is defined as **raw heads**: l/t/r/b distances in grid units plus class logits (the same
contract as YOLO26). The packs produced by `tools/surgery` (and by SiMa's `model_surgery`)
carry **decoded** `cx,cy,w,h` pixel boxes, declared as `bbox_format: cxcywh_pixel` in
`boxdecoder.json`. 0.6.0 ignores that field and treats the pixel coordinates as distances.
There is no runtime switch: `decode_type_option` only changes head grouping, and the generic
`yolo` (decoded) family is rejected for model packs. Tracked as
[sima-neat/core#1272](https://github.com/sima-neat/core/issues/1272).

**Fix (model side, no runtime change).** Build the pack with raw heads and tell the decoder the
head layout:

```bash
python3 tools/yolov6_raw_export/yolov6_raw_export.py my_yolov6.onnx my_yolov6_raw.onnx
# quantize + compile my_yolov6_raw.onnx with ModelSDK 3.0.0 as usual (int8 or bf16)
```

```yaml
models:
  - {name: my-v6, path: ../models/my_yolov6_raw_mpk.tar.gz,
     decode_type: yolov6, decode_type_option: GroupedByRoleLogit, num_classes: 80,
     labels: my_labels.txt, min_score: 0.30}
```

Both extra fields are required. `GroupedByRoleLogit` describes the output order the script
emits (3 bbox heads, then 3 class-logit heads); `num_classes` is needed because a raw pack has
no `boxdecoder.json` to derive it from. The decode math is unchanged, only moved from the graph
into the decoder, so accuracy and thresholds carry over. Decoding stays on the built-in
decoder, so there is no CPU cost compared with the old packs.

## Verification

Tested on two Modalix DevKits on platform 3.0.0 B2501: one with the 0.6.0 release runtime,
one with `0.6.0+main.b2001ee23b8c`. Same YOLOv6s model, same 720p highway frame, PyNeat
`Model` API, `score 0.30 / nms 0.60 / top_k 50`.

| pack | decode_type | decode_type_option | result |
|---|---|---|---|
| decoded heads (`tools/surgery`) | yolov6 | Auto | 50 boxes, 18 of them `(0,0)-(1280,720)`, all starting at the origin |
| decoded heads | yolov6 | GroupedByRole / GroupedByRoleLogit / GroupedByRoleProbability | identical to Auto |
| decoded heads | yolo | any | rejected: "The MPK did not declare a concrete BoxDecode decode_type" |
| raw heads (`tools/yolov6_raw_export`) | yolov6 | Auto | fails to build: "logical channel count exceeds source storage" |
| raw heads | yolov6 | **GroupedByRoleLogit** | **13 tight boxes on the vehicles, scores up to 0.82** |
| raw heads | yolov6 | GroupedByRoleLogit, no `num_classes` | fails to build: "class count must be 1..336" |

Independent re-run (2026-11-03, fresh export + fresh bf16 compile + fresh board run on the 0.6.0 release
board): same outcome. Export tool accepts both the official export and the surgered ONNX (identical raw
outputs). Against an onnxruntime run of the original YOLOv6s ONNX with the same preprocessing, 10 of the 13
board boxes match the host decode at IoU > 0.5, 9 of them within 5 px; the rest are low-score edge cars the
host scores just under 0.30 and one NMS near-duplicate. Scores run up to +0.2 higher on the board (bf16 pack,
different resize kernel). 3-class DFL pack: DFL projection matches the ONNX math to 3e-6; on-device boxes
within 1 px of the host reference decode.

Raw tensor ranges confirm the diagnosis: the decoded pack's bbox heads span 2..648 on a 640
input (pixels); the raw pack's span -0.1..10 (grid-unit distances).

### 42 streams @ 3 fps (the field configuration this was reported from)

SiMa's high-density multi-stream detector (Python, apps `main` 2026-11-01, with `yolov6` +
`decode_type_option` added), one shared YOLOv6s bf16 raw-head pack, 42 × 720p H.264 RTSP
sources at 3 fps, metadata to a UDP collector:

| metric | value |
|---|---|
| channels delivering detections | 42 / 42 |
| per-channel frame rate at the collector | min 3.0, max 6.0 (loop bursts), mean 3.8 /s |
| boxes per frame (mean) | 4.1 |
| A65 load | ~1.2 cores of 16 (python process), >90 % idle |
| CMA | 8.7 GB free of 10 GB, flat |
| graph build | 3.4 s |

### int8
The int8 compile of the same raw 3-class ONNX (bf16 flags dropped) places identically (MLA 1 / EV74 15 / A65 0),
takes 10.5 min, is half the size (15.4 vs 31.5 MB) and runs 9.5 ms vs 15.2 ms per inference incl. preprocess and
decode on the DevKit. On the COCO test image all 12 int8 boxes match a host DFL reference decode within 8 px
(mean 3.6 px); scores are ~0.03 lower than bf16 on average and 4 low-score (<=0.41) bf16 boxes drop out.

## DFL variants (`reg_max > 0`)

YOLOv6 heads trained with DFL have `4 × (reg_max+1)` regression channels (68 for
`reg_max=16`). The 0.6.0 decoder wants 4 channels, so the projection
(softmax over bins, expectation) must stay in the graph. `yolov6_raw_export.py` adds it
automatically and matches the reference math to 1e-6. Not every formulation compiles: the
ModelSDK only accepts Softmax on what it sees as the channel axis, and a
Transpose→Softmax(axis=1)→Conv form is rejected ("Softmax is supported only along channel axis")
and then fails in the MLA simulator. The tool therefore emits
`[1,4,HW,bins]` → Softmax(axis=3) → MatMul(`0..reg_max`) (default), or with `--dfl exp` a
softmax-free Exp → two grouped 1×1 Convs → Div. Both compile with every op on the MLA
(ModelSDK 3.0.0 bf16: MLA 1 / EV74 15 / A65 0).

Verified on a 3-class `reg_max=16` YOLOv6 on the DevKit (0.6.0 release): with
`yolov6` + `GroupedByRoleLogit` + `num_classes: 3`, boxes match a host-side reference decode of
the original ONNX to within 1 px (e.g. person 0.94 at (594,24)-(640,252) vs (594,25)-(640,253)).

## Notes

- ModelSDK 3.0.0's `afe.apis.defines` no longer exports `gen1_target`; older compile scripts
  must import it optionally.
- Do not run `tools/surgery` after the raw export; that step folds the decode back in.
- In the `ghcr.io/sima-neat/sdk:latest` container the compiler shells out to a bare `mla-masm`, which is not on
  PATH even in a login shell; run the compile with `export PATH=/sdk-extensions/model-compiler/bin:$PATH` or
  it fails late in code generation with `FileNotFoundError: 'mla-masm'`. The `sdk:v3.0.0.0` image cannot
  `import afe` at all (missing `libLLVM.so.18.1`).
- `num_classes` is mandatory with the raw pack (see table); the raw pack has no class count metadata the decoder trusts.
- The C++ app in `som/` accepts `decode_type_option` and per-model `num_classes` on this branch
  (it needs the Neat 0.6.0 headers to build; older sysroots lack `decode_type_option`). The SDK 3.0.0
  container sysroot ships `libyaml-cpp.so.0.8` but not its headers: copy `/usr/include/yaml-cpp` from a
  board with `libyaml-cpp-dev` installed, or add `-I` to a local copy. Syntax-checked against the
  3.0.0 sysroot with the aarch64 GCC 16 toolchain (`-idirafter <sysroot>/usr/include/aarch64-linux-gnu`
  is needed for `bits/wordsize.h`).
