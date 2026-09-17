# Modalix Multi-Stream, Multi-Model Detector

High-density RTSP object detection on a **single SiMa Modalix SoM** — up to
**48 camera streams** through a **shared multi-model detector** with **per-camera
model selection at runtime**, plus host-side apps for live viewing and object
cropping.

- **On-device (`som/`)** — one fused Neat graph. Every camera fans into a small
  set of detectors held warm on the MLA. Switch a camera's model over HTTP and
  the graph re-wires in ~2 s — no model reload.
- **Host (`host/`)** — a live viewer (HLS video + detection overlays), a
  content-aware crop app (saves cut-outs of every detection across all 48
  streams), and a control panel to pick the model and crops per camera.

> Derived from the Apache-2.0 [`sima-neat/apps`](https://github.com/sima-neat/apps)
> `high-density-multi-stream-object-detector` example. See `NOTICE`.

---

## Why this shape

The single MLA is the shared resource. The design that scales is **one detector
per _model_, not per _camera_** — all cameras that want model *M* fan into the
one shared *M* detector via a realtime "latest-frame-per-stream" link. So *K*
models cost *K* detectors regardless of how many of the 48 cameras route to
them.

```
48 RTSP ─► H.264 decode ×48 ─► fan-in ─► shared detectors ─► results
                                          (coco-n │ coco-v8 │ coco-m)
                                                  ▲
                                POST /api/streams/{id}/models  (runtime switch)
```

Two ways to run more than one model (both supported per camera):

| Mode | Meaning | Cost |
|---|---|---|
| **Route** | each camera → **one** model (different cameras, different models) | 1 inference/frame — **full rate** |
| **Chain** | each camera → **several** models, boxes merged | N inferences/frame — scales to ~half the streams |

A block diagram and a step-by-step of the runtime switch are in
[`docs/model-switching.html`](docs/model-switching.html) and
[`docs/MODEL_SWITCHING.md`](docs/MODEL_SWITCHING.md).

---

## Repository layout

```
som/                 on-device (Modalix SoM) detector
  src/cpp/           main.cpp + CMakeLists.txt  (C++, Neat API)
  configs/           example YAML configs
  scripts/launch.sh  detached launcher
host/                host-side web apps (Python stdlib + OpenCV + PyAV)
  viewer/            serve.py + index.html   — HLS video + box overlays (:8090)
  crops/             crops_host48.py + gallery.html + control.html (:8092)
  systemd/           user-service unit files
models/              (empty) — bring your own compiled packs; see models/README.md
docs/                architecture, model-switching, benchmarks, diagram
```

---

## Quickstart

### 1. Build the on-device detector

The detector uses the Neat C++ API and builds inside the SiMa Neat SDK
container (it links the `sima_neat_apps_support_*` libraries and the Modalix
sysroot). Drop `som/src/cpp/` into a checkout of `sima-neat/apps` as a new
example, or build against your Neat SDK toolchain:

```bash
# inside the Neat SDK container, with the Modalix toolchain on the sysroot
cmake -S som/src/cpp -B som/build \
      -DCMAKE_TOOLCHAIN_FILE=<neat-apps>/cmake/toolchains/aarch64-modalix.cmake
cmake --build som/build -j
```

Put your compiled model packs in `models/` (see [`models/README.md`](models/README.md)).

### 2. Run it on the SoM

```bash
# on the board, next to the built binary + models/
./scripts/launch.sh configs/example-48x5fps.yaml
curl http://localhost:8600/api/streams        # 48 streams, live fps
```

Other example configs in `configs/`:
- `example-12x30fps-crops.yaml` — 12 streams @ 30 fps with on-SoM exact crops.
- `example-yolov6-multimodel.yaml` — YOLOv6 + YOLO26 loaded together, cameras
  routed per model (24 streams @ 5 fps).
- `example-yolov6-dual.yaml` — YOLOv6n loaded as **two instances** with the
  cameras split across them, reaching the MLA ceiling of **~38 streams @ 5 fps**
  (~189 fps, measured on-board; vs ~30-36 on a single YOLOv6n instance — see
  [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).

### 3. Run the host apps

```bash
export SOM_API=http://<SOM_IP>:8600
python3 host/viewer/serve.py        # live viewer   -> http://<host>:8090
python3 host/crops/crops_host48.py  # crops + panel -> http://<host>:8092
#                                        control panel -> http://<host>:8092/control
```

(Or install the `host/systemd/*.service` units for boot persistence — set the
`SOM_API` IP first.)

---

## HTTP control API (`:8600`)

| Method + path | Does |
|---|---|
| `GET /api/streams` | per-stream url, models, fps, mode |
| `GET /api/models` | loaded models + which are active |
| `GET /api/results[/<n>]` | latest detections (bbox, label, conf, `pts_ns`) |
| `POST /api/streams/<n>/models` | set one camera's model(s) — `{"models":["coco-m"]}` |
| `POST /api/models_all` | set every camera to a model |

The host apps proxy these, so the browser talks to one origin.

---

## Notes & limits

- **INT8 vs BF16** is a *throughput* choice, not a memory one — both use ~the
  same CMA; INT8 is ~2× faster on the MLA. The stream ceiling is driven by
  **decoder frame buffers** (`max_inflight_per_stream`), not model precision.
  See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
- The accurate BF16 model sustains ~6 cameras on one MLA; the control panel
  warns past that.
- Exact on-SoM crops (`crops.enabled`) fit a ~12-camera subset; the host crop
  app covers all 48 by matching each detection's `pts_ns` to a locally decoded
  frame (content-aware selection for fast movers).

## Docker overlay demo (one container per stream)

[docker-overlay/](docker-overlay/) runs one Docker container per camera stream, draws the detection overlay on the
SOM and sends hardware-encoded video to the viewer. 16 containers ran for 18 hours under fault injection with 99.93%
video availability, and a buffer-reuse fix then removed the self-relaunches seen in that run (8 h, 100% availability). Grouping 4 to 8
streams per container raises the ceiling from 18 to 28 streams. See [docker-overlay/README.md](docker-overlay/README.md).

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Not an official
SiMa release; model packs and the Neat SDK are not included.
