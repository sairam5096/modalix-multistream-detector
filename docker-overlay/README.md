# Docker overlay demo: one container per camera stream, overlay drawn on the SOM

Each camera stream runs in its own Docker container on the Modalix SOM. The container decodes the stream, runs YOLO
on the MLA, draws the boxes **on the SOM**, re-encodes with the hardware H.264 encoder and sends the finished video
to a viewer (Neat Insight). Streams are sandboxed from each other, a crashed stream is relaunched by Docker, and
fps, resolution and model can be changed per stream at run time by re-creating that one container.

## At a glance

| Item | Value |
|---|---|
| Streams per container | 1 by default, several with repeated `--url` |
| Stable, soak-proven | 16 containers of 1 stream, 720p at 5 fps, 8 h under fault injection |
| Most streams on one DevKit | 28, with 4 or 8 streams per container (3-minute check only) |
| Soak 1, first version | 18 h 24 min, fault every 20 min: 99.93% availability, but about 6 self-relaunches per hour |
| Soak 2, buffer-reuse fix | 8 h 05 min, fault every 20 min: 100% availability, 0 allocation errors, 0 unplanned restarts |
| Injected faults recovered | 47 of 47 in soak 1, 21 of 21 in soak 2, neighbours never affected |
| Per-frame app cost (C++) | about 7.5 ms (NV12 to BGR 3 ms, draw 1.6 ms, push 2.8 ms) |
| Per-container footprint | about 230 MB RAM, about 45 MB pinned CMA plus 19 MB codec memory, about 45% of one core |
| Limits | one stream per container: RAM at 18. Grouped: CMA and CPU at 28 |

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
| `test/combo_sweep.sh`, `test/combo_matrix.sh` | find the most total streams for K streams per container, reboot between K values |
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

Several streams in one container: repeat `--url`. Channels count up from `--channel`.

```
./run_overlay.sh 1 rtsp://MEDIA_HOST:8554/cam01 0 --fps 5 --url rtsp://MEDIA_HOST:8554/cam02 --url rtsp://MEDIA_HOST:8554/cam03
```

Each stream is an independent pipeline (own decoder, model session, encoder, buffer ring) in one process. If any
pipeline ends, the process exits and Docker relaunches the whole container.

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
| `--push-pool` | 4 | ring of reusable encoder input buffers, 0 allocates one per frame (old behaviour) |
| `--alloc-retry-ms` | 1000 | how long to retry a failed buffer allocation before dropping that frame |
| `--encoder` | hw | `hw` = SiMa H.264 encoder, `sw` = x264 on the CPU, no CMA used on the egress side |
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

## Results

### Soak 1: first version, 16 containers, 720p at 5 fps, 18 h 24 min

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

### The problem soak 1 exposed, and the fix

```
 CMA region (about 1.8 GB)
 +----------------------------+---------------------------+-----------+
 | pinned video/ML buffers    | page cache (movable)      | free      |
 | about 750 MB, flat         | about 1000 MB             | 40-85 MB  |
 +----------------------------+---------------------------+-----------+
   new encoder buffer needed -> kernel must move cache pages out -> sometimes too slow
   -> allocation fails -> app exits -> Docker relaunches it in about 8 s
```

The first version allocated a fresh hardware buffer for every frame. At 16 containers that allocation failed about
6 times per hour, the app exited and Docker relaunched it. It was not a leak, pinned buffers stayed flat.

```
 before:  every frame -> allocate new CMA buffer -> copy image in -> push to encoder -> free
                              ^ fails when CMA is full -> exception -> app exits

 after:   start-up    -> allocate 4 buffers once
          every frame -> pick next of the 4 -> map, copy image in -> push to encoder
          fallback    -> retry allocation for up to 1 s, then drop that one frame, never exit
```

The ring is filled through `Tensor::map_write()`. The encoded output was captured on the host and checked for tearing
and stale frames: none. Push cost dropped from 4.3 ms to 2.8 ms per frame.

### Soak 2: with the fix, 16 containers all on YOLOv6n, 720p at 5 fps, 8 h 05 min

Started after a clean reboot.

| Metric | Soak 1 (before the fix) | Soak 2 (after the fix) |
|---|---|---|
| Containers up | 16 of 16 | 16 of 16 at all 441 samples |
| CMA allocation errors | about 6 per hour | 0 |
| Allocation retries / dropped frames | not applicable | 0 / 0 |
| Unplanned relaunches | about 100 | 0 |
| Video availability | 99.93% | 100% on every channel |
| Injected faults recovered | 47 of 47 | 21 of 21 |
| RAM in use | 4.7 to 5.1 GB | 5.13 to 5.21 GB, flat |
| Pinned CMA | about 460 MB | 648 to 666 MB, flat (the buffer rings stay allocated) |
| SoC temperature | 46 to 52 °C | 50 to 54 °C |

All 10 relaunches in soak 2 were caused by the test: 4 injected crashes, 4 camera outages, and 2 decoder failures at the
unpause moment of the 20 s freeze action. After a 20 s process freeze the hardware decoder re-opens and can fail to
re-initialise, the app exits and is streaming again within 9 s. One of three freezes survived without a relaunch.

Tip: the board clock can drift from the host clock (7 min 48 s here). Apply the offset before correlating container
logs with harness events.

### Hardware encoder or CPU encoder

`--encoder sw` encodes with x264 on the CPU and uses no CMA on the egress side. One 720p 5 fps stream, measured next
to 15 other running containers:

| | Hardware encoder | CPU encoder (x264) |
|---|---|---|
| Container CPU, % of one core | 43 to 45 | 60 to 62 |
| Container RAM | 231 MB | 287 MB |
| Pinned CMA plus codec memory | 64 MB | 42 MB |
| Output | clean | clean, slightly softer |

Each CPU-encoded stream saves about 22 MB of CMA and costs about 16% of a core and 56 MB of RAM. RAM is the next
limit, so use it as a fallback for a few streams, not for all of them. Power was not measured, the DevKit has no
power sensor.

## How many streams per container

Same app, YOLOv6n, 720p at 5 fps, clean reboot before each group size, containers added until a stop rule tripped,
then a 3-minute steady check.

```
 K=1 :  [s1] [s2] [s3] ...            best isolation
 K=4 :  [s1..s4] [s5..s8] ...
 K=8 :  [s1..s8] [s9..s16] ...        fewest processes, a failure restarts 8 streams
```

| Streams per container | Containers | Max total streams | CPU idle at max | Limit hit |
|---|---|---|---|---|
| 1 | 18 | 18 | 31 to 42% | RAM, about 250 MB per container |
| 2 | 13 | 26 | about 11% | RAM, with CPU close behind |
| 4 | 7 | **28** | about 9% | CMA and CPU |
| 8 | 3, plus one of 4 | **28** | 9 to 12% | CMA and CPU |
| 16 | 1, plus one of 8 | 24 | 20% | a second group of 16 does not fit |

- Grouping cuts RAM per stream from about 250 MB to about 55 MB, because streams share one process. RAM stops being the
  limit and CMA plus CPU take over.
- At 28 streams the pinned hardware buffers total about 1690 MB of the 1788 MB CMA region, about 60 MB per stream.
- **Overload is not graceful.** Twice, a failed start beyond capacity (towards 32 streams) left the whole board degraded:
  no video on any channel and 84% system CPU time, cleared only by a reboot. Cap the stream count below the CMA ceiling.
- 28 is a ceiling from a 3-minute check, not a soak-proven operating point. 24 streams (3 containers of 8, or 6 of 4)
  leaves about 20% CPU idle and is the suggested target, pending its own soak.
- The first container after a reboot usually fails its first RTSP connect and is relaunched once by Docker. A sweep rule
  must not count that as a failure.

| | 1 per container | 4 to 8 per container |
|---|---|---|
| Max streams | 18 | 28 |
| Blast radius of one failure | 1 stream | 4 to 8 streams restart together |
| Status | 16 proven over 8 h | ceiling measured, soak pending |

## Notes

- Results were measured on a Modalix DevKit with a pre-release eLxr 3.0 / Neat build and a kernel with namespace
  support. Older board kernels without `CONFIG_NAMESPACES` cannot run Docker containers.
- A YOLOv6 pack must emit its outputs box tensors first, then class tensors (`[4,4,4,C,C,C]`) for the in-graph
  YoloV6 decode. A class-first pack decodes to point-sized boxes.
- The same pipeline in Python (PyNeat) topped out at 12 containers on the same board.
