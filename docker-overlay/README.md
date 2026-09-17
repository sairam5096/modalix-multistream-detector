# Docker overlay demo: one container per camera stream, overlay drawn on the SOM

Each camera stream runs in its own Docker container on the Modalix SOM. The container decodes the stream, runs YOLO
on the MLA, draws the boxes **on the SOM**, re-encodes with the hardware H.264 encoder and sends the finished video
to a viewer (Neat Insight). Streams are sandboxed from each other, a crashed stream is relaunched by Docker, and
fps, resolution and model can be changed per stream at run time by re-creating that one container.

## At a glance

| Item | Value |
|---|---|
| Streams per container | 1 |
| Containers tested on one DevKit | 16 at 720p, 5 fps |
| Soak | 18 h 24 min with a fault injected every 20 min |
| Video availability | 99.93% overall, worst channel 99.80% |
| Injected faults recovered | 47 of 47, neighbours never affected |
| Per-frame app cost (C++) | about 9 ms (NV12 to BGR 3 ms, draw 1.6 ms, push 4.3 ms) |
| Per-container footprint | about 280 MB RAM, about 47 MB pinned CMA, about 37% of one core |
| Limit at 16 | contiguous memory (CMA), see "Known limit" |

## What one container does

```
 IP camera        +------------------- one Docker container ---------------------+        Neat Insight
 (RTSP, 720p) --> | HW decode -> YOLO on MLA -> draw boxes (OpenCV) -> HW encode | --RTP-->  viewer
                  +---------------------------------------------------------------+        (1 channel)
```

Inside the app (`app/main.cpp`):

```
 RtspEncodedInput -> SimaDecode (NV12) -> Branch --+--> Model (CVU preprocess + MLA + in-graph box decode) -> Output "detections"
                                                   +--> Output "frame"
 app thread: pair by frame_id -> NV12 to BGR -> cv::rectangle / putText
          -> Tensor::from_cv_mat(BGR, EV74) -> Input -> VideoSender (H.264 HW encode, RTP/UDP)
```

## How many share one SOM

```
 cam 01 --> [ container 1  ] --> viewer ch 0
 cam 02 --> [ container 2  ] --> viewer ch 1
   ...          ...                 ...
 cam 16 --> [ container 16 ] --> viewer ch 15
            shared hardware: video decoder | MLA | video encoder | RAM | CMA
            Docker restart policy relaunches any container that exits
```

The image is `FROM scratch` and is 0 bytes. All runtime files (Neat libraries, GStreamer plugins, the app binary,
models) are read-only bind mounts from the board, so 16 containers add no storage and share the page cache.

## Layout

| Path | Purpose |
|---|---|
| `app/` | C++ source, CMake project, cross-build script |
| `app/support/object_detection/` | box parsing helpers from [sima-neat/apps](https://github.com/sima-neat/apps), Apache-2.0 |
| `docker/Dockerfile` | the empty image |
| `docker/run_overlay.sh` | start one container: `./run_overlay.sh N RTSP_URL CHANNEL [app args] [-- docker args]` |
| `docker/recreate.sh` | change parameters of one running stream (model, fps, resolution) |
| `docker/demo.env.example` | addresses and paths, copy to `demo.env` |
| `test/scale_sweep.sh` | add containers one at a time until a health rule trips |
| `test/endurance.sh` | unattended soak with fault injection and full log capture |

Model packs are not shipped. Put the default pack at `docker/models/yolo26n-det-int8-b1.tar.gz` and any extra packs
in `MODELS_DIR`.

## Build

Inside the SiMa Neat SDK container, which has the aarch64 sysroot:

```
cd docker-overlay/app && ./build.sh
```

Copy `app/build/overlay-detector` to `docker/build/overlay-detector` on the board.

## Run on the board

```
cd docker-overlay/docker
cp demo.env.example demo.env        # set INSIGHT_HOST and MEDIA_HOST
docker build -t neat-overlay:mounted .
./run_overlay.sh 1 rtsp://MEDIA_HOST:8554/cam01 0 --fps 5
./run_overlay.sh 2 rtsp://MEDIA_HOST:8554/cam02 1 --fps 5 --decode yolov6 --model models2/yolov6n_mpk.tar.gz
```

Start containers one at a time with a gap of about 15 s. The MLA does not like rapid load and unload cycles.

App arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--url`, `--channel`, `--host` | required | RTSP source, viewer channel, viewer address |
| `--fps`, `--width`, `--height` | 5, 1280, 720 | source format |
| `--model`, `--decode`, `--labels` | yolo26n int8, `yolo26`, `labels.txt` | decode is one of yolo26, yolov8, yolov6, yolov5, yolov10 |
| `--bitrate`, `--min-score` | 2000 kbps, 0.30 | encoder bitrate, score threshold |
| `--dec-bufs`, `--dec-in-bufs`, `--dec-tuning`, `--dec-memopt` | 4, 2, low-memory, 1 | lean decoder pools, the settings that made 16 fit |
| `--out-q`, `--mla-pool` | 2, runtime default | output queue depth, MLA output pool |
| `--stall-exit-s` | 60 | exit when no frames arrive, so Docker relaunches the stream |
| `--save-frame` | off | write one annotated JPEG for a visual check |

## Test method

```
 Host harness (every 60 s) ---- reads ----> container states, restart counts, RAM, CMA, temp, kernel log
        |                 \---- reads ----> viewer video rate per channel (health signal)
        +-- every 20 min, one random container gets one of:
              restart | kill -9 | camera outage 45 s | change model / fps / resolution | freeze 20 s
        +-- watchdog: channel silent for 3 samples -> restart that container
```

```
export BOARD=SET-SOM-IP BOARD_PASS=... INSIGHT=https://SET-HOST-IP:PORT
STREAM_FMT='rtsp://SET-HOST-IP:8554/cam%02d' test/scale_sweep.sh 24 15 1
CHAOS=1 CHAOS_EVERY_MIN=20 setsid nohup test/endurance.sh run &
test/endurance.sh report
test/endurance.sh stop
```

The camera outage action freezes the ffmpeg publisher of the chosen stream, so it only works when the harness runs
on the machine that publishes the streams.

## Results: 16 containers, 720p at 5 fps, 18 h 24 min

12 containers ran YOLO26n int8 and 4 ran YOLOv6n.

| Metric | Result |
|---|---|
| Containers up | 16 of 16 at every sample, 0 exited |
| Board unreachable / Docker unresponsive | never |
| New kernel errors | 0 |
| Video availability | 99.93% overall, worst channel 99.80% |
| Watchdog interventions | 0 |
| RAM in use | 4.7 to 5.1 GB of 5.9 GB, flat |
| SoC temperature | 46 to 52 °C |
| Docker auto-relaunches | 116, of which 8 were injected crashes |

| Fault injected | Count | Recovered | Neighbours affected |
|---|---|---|---|
| Graceful restart | 8 | 8 | 0 |
| Process crash (kill -9) | 8 | 8, auto-relaunch | 0 |
| Camera outage, 45 s | 8 | 8 | 0 |
| Live change of model, fps or resolution | 15 | 15 | 0 |
| Freeze, 20 s | 8 | 8 | 0 |
| **Total** | **47** | **47** | **0** |

## Known limit: CMA at 16 containers

```
 CMA region (about 1.8 GB)
 +----------------------------+---------------------------+-----------+
 | pinned video/ML buffers    | page cache (movable)      | free      |
 | about 750 MB, flat         | about 1000 MB             | 40-85 MB  |
 +----------------------------+---------------------------+-----------+
   new encoder buffer needed -> kernel must move cache pages out -> sometimes too slow
   -> allocation fails -> app exits -> Docker relaunches it in about 8 s
```

At 16 containers a per-frame DMA-BUF allocation fails about 6 times per hour. The app exits and Docker relaunches it
in about 8 s. It is not a leak, pinned buffers stayed flat for the whole run. Expect about 14 containers to run
without self-relaunches. Planned fixes:

| Fix | Effort | Where |
|---|---|---|
| Reuse the encoder buffer, or retry on failure instead of exiting | small | `app/main.cpp` |
| Drop the page cache periodically | small | board cron job |
| Enlarge the CMA reservation | medium | device tree, needs reboot |

## Notes

- Results were measured on a Modalix DevKit with a pre-release eLxr 3.0 / Neat build and a kernel with namespace
  support. Older board kernels without `CONFIG_NAMESPACES` cannot run Docker containers.
- A YOLOv6 pack must emit its outputs box tensors first, then class tensors (`[4,4,4,C,C,C]`) for the in-graph
  YoloV6 decode. A class-first pack decodes to point-sized boxes.
- The same pipeline in Python (PyNeat) topped out at 12 containers on the same board.
