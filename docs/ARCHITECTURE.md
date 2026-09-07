# Architecture

## On-device: one fused graph, shared detectors

The detector builds a **single Neat graph** for all cameras. Per camera:

```
RTSP source → H.264 depay → SiMa H.264 decoder (NV12, zero-copy) → fan-in link → shared detector → box decode → results
```

- **Shared detectors.** One detector stage is built per model that is *in use*.
  Every camera routed to model *M* connects to the same *M* stage through a
  fan-in link with policy `RealtimeLatestByStream` and a per-camera
  `stream_id`, so the stage always processes the newest frame of each stream and
  results are demuxed back per camera.
- **Warm registry.** All 2–4 configured models are loaded once at startup and
  kept warm. Only the ones a camera currently routes to are built into the
  graph.
- **Route vs. chain.** A camera carries a *set* of models. One model = route;
  several = chain (each model runs on the frame, boxes merged and labelled by
  model).

### Memory: what actually costs CMA

CMA (contiguous DMA memory) is the tight resource. The dominant consumer is the
number of **decoded frames held in flight per stream**, set by
`inference.max_inflight_per_stream` and the decoder buffer counts:

- Keep `max_inflight_per_stream` at **1–2** and `decoder_tuning: low-memory` for
  high stream counts.
- Model **precision (INT8 vs BF16) does not materially change CMA** — see
  `BENCHMARKS.md`.

### Self-healing

A pump thread pulls detections from each detector's output. A watchdog thread
monitors throughput and per-stream progress and rebuilds the graph if it stalls
or collapses (with a post-build grace period so the ramp-up of many streams is
not mistaken for a stall). Model switches are **coalesced**: a rebuild request
is debounced ~0.6 s so a burst of switches becomes one rebuild.

## Host side

Two independent Python apps (stdlib HTTP server; OpenCV + PyAV for the crop
app). They read the SoM over the control API and never decode on the SoM's
behalf.

- **Viewer (`:8090`)** — serves a page that plays each camera as **HLS video**
  (straight from your RTSP/HLS server) and draws detection boxes from
  `/api/results` on a canvas overlay. A "sync" control delays the boxes to line
  up with the buffered video. Model buttons POST to the control API.
- **Crop app (`:8092`)** — decodes each unique source with PyAV, polls
  `/api/results`, and saves a JPEG cut-out of every detected object. Because the
  host decoder and the SoM decoder are independent, it recovers a **per-stream
  constant time offset** between them and then does **content-aware frame
  selection** (picks the candidate frame where the boxes have the most in-box
  motion) so fast objects land in-frame. A gallery shows newest-first crops.
- **Control panel (`:8092/control`)** — select cameras, then bulk-apply a model
  (proxied to the SoM) or toggle crops on/off (host-side, instant). Warns when
  more than ~6 cameras are put on the accurate BF16 model.

```
                         ┌──────────── Modalix SoM ────────────┐
 48 RTSP  ──►  decode ×48 ─► fan-in ─► shared detectors ─► results ──► host apps
   │                                    ▲                                 │
   │ HLS video                          │ rebuild (coalesced)             │ /api/results
   └───────────────────────────────►  Control API :8600  ◄───────────────┘
                                        ▲  POST /api/streams/{id}/models
                                        └── host control panel (:8092/control)
```
