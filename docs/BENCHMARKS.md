# Benchmarks & findings

Measured on one SiMa Modalix SoM (16× Cortex-A65, single MLA, ~1788 MB CmaTotal,
SDK 2.1.3 / Neat 0.4.0). Sources: 720p H.264 RTSP. Steady-state after all streams
bind. `coco-n` = YOLO26n INT8, `coco-v8` = YOLOv8n INT8, `coco-m` = YOLO26m BF16.

## 1. Topology: route vs. chain (48 streams @ 5 fps)

| Topology | Active | Total fps | fps/stream | CMA used |
|---|---|---|---|---|
| Single shared model | 48/48 | 239 | 5.0 | 1319 MB |
| **Route** (24 model-A + 24 model-B) | 48/48 | 239 | 5.0 | 1495 MB |
| **Chain** (both models on every camera) | ~42/48 | ~60 | ~1.25 | 1560 MB |

**Routing distinct cameras to distinct models does not degrade** — it runs at the
full single-model rate. Chaining runs every frame through both models (2×
inferences + 2× working memory), so it holds only ~half the streams. Prefer
route; use chain only where a camera genuinely needs two models.

## 2. Precision: INT8 vs BF16 (48 streams, identical decoder settings)

| Precision | Active | Total fps | fps/stream | CMA used |
|---|---|---|---|---|
| INT8 | 48/48 | 239 | 4.98 | 1319 MB |
| BF16 | 48/48 | 130 | 2.71 | 1327 MB |

- **CMA is essentially identical (1319 vs 1327 MB)** — precision does *not* drive
  memory. BF16 runs 48 streams fine; it is not memory-bound.
- The difference is **throughput**: the BF16 MLA path is ~2× slower.

So switching INT8↔BF16 is a **speed** trade-off, not a way to fit more streams.
Accuracy (COCO box mAP, YOLO26n): **BF16 0.379 vs INT8 0.342** (~3.7 pts);
single-stream MLA throughput ~262 fps (BF16) vs ~559 fps (INT8).

## 3. The real stream-count lever: decoder frame buffers

The stream ceiling is hit as `cma_alloc failed, req-size: 360 pages` — 360 pages
× 4 KB = 1.44 MB = **one 720p NV12 frame buffer**. The limit is how many decoded
frames are held **in flight per stream**:

| `max_inflight_per_stream` | CMA per stream |
|---|---|
| 4 | ~65 MB |
| 2 | ~27 MB |

Keep it at **1–2** with `decoder_tuning: low-memory` and small `decoder_buffers`
to maximize stream count. This is independent of how you read RTSP (e.g. ffmpeg
vs the Neat source) — the frames still land in CMA for the MLA.

## 4. Crops

- **On-SoM exact crops** (same decoder as detection, matched by `pts_ns`) are
  pixel-perfect but fit a **~12-camera** subset on the single MLA.
- **Host-side crops** cover **all 48**: the host decodes each source
  independently and recovers a per-stream constant time offset, then uses
  content-aware frame selection (max in-box motion among candidate frames) so
  fast objects are caught. Accurate for the large majority; a fast object at
  5 fps can still be ~1 frame off.

## 5. YOLOv6 throughput ceiling

YOLOv6n runs in the shared multi-model framework via `decode_type: yolov6` (see
`som/configs/example-yolov6-multimodel.yaml`) and detects correctly, but its MLA
path is slower than YOLO26n, so it caps the stream count:

| Model | MLA throughput (this SoM) | Usable @ 5 fps |
|---|---|---|
| YOLO26n INT8 | ~240 fps | 48 streams (240 fps demand) |
| YOLOv6n INT8 | ~187 fps | ~30-36 streams (≤ ~180 fps demand) |

Past the throughput ceiling the failure is **backpressure, not a leak**: decode
keeps producing at the source rate while inference lags, so decoded frames pile
up in CMA (measured drain ~7 MB/s at 48 streams) until the pool is exhausted and
the board reboots. Verified by holding load below the ceiling — 12 streams
(60 fps) and 30 streams (150 fps) run with CMA flat for minutes; only demand
above ~187 fps drains it.

**Two YOLOv6n instances reach ~38 streams — the MLA ceiling, not more.** Loading
YOLOv6n **twice** as two model instances and splitting the cameras across them
lets it stably serve **~38 streams @ 5 fps (~189 fps)**, up from ~30-36 with a
single instance. The two instances give the MLA two independent pipelines to
interleave, recovering the last bit of scheduling headroom a single instance
leaves idle — but that is a pipelining gain, not extra compute, so it only takes
throughput up to the aggregate MLA ceiling (~187-190 fps) and no further. Measured
on-board (2026-09-10, two YOLOv6n instances, 43-stream oversubscribed config): a
stable **38/43 streams active at 188.6 fps**, CMA flat and 0 rebuilds over several
minutes; the 5 overflow streams stay demand-starved at 0 fps because aggregate
demand (215 fps) exceeds the ceiling. So the second instance is a modest, real
lift for a YOLOv6-only deployment — enough to reach the MLA ceiling — but it does
**not** reach 43 streams (see `som/configs/example-yolov6-dual.yaml`).

**Other ways to run YOLOv6 at high density**: fewer streams, a lower source fps
(decimate at the source, *not* via `target_fps`, which adds a CMA-draining
`videorate`), or route only some cameras to YOLOv6 and the rest to a faster model
(as `example-yolov6-multimodel.yaml` does).

## Takeaways

1. **Route, don't chain** for multiple models across cameras.
2. To raise the stream ceiling, cut **in-flight decoder buffers**, not precision.
3. **INT8** for throughput headroom; **BF16** for ~3.7 mAP more accuracy on a
   handful of cameras.
4. **YOLOv6n** works but tops out around **30-36 streams @ 5 fps** (~187 fps MLA)
   with one instance; loading it as **two instances** and splitting the cameras
   reaches the MLA ceiling of **~38 streams (~189 fps, measured on-board)** — a
   modest lift, not the ~43 an oversubscribed config appears to ask for. Keep
   aggregate demand under the ceiling or route the overflow to a faster model.
