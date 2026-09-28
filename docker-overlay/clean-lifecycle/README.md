# Clean start-up and clean shutdown for a YOLOv6s container

A sample that starts a YOLOv6s detector container, confirms it is healthy, stops it
quickly and in order, and proves with a cycle test that nothing stays allocated
between runs.

```
 start  ->  wait for "detector running"  ->  run  ->  docker stop (SIGTERM)
                 |                                         |
                 v fails                                   v
        show why, remove container              app closes pipelines, exit 0 in < 1 s
```

## What it does and why

| Rule | Without it | With it |
|---|---|---|
| The app handles `SIGTERM` / `SIGINT` | The app is PID 1 in its container, so it ignores `SIGTERM`. Docker waits the whole stop timeout, then kills it: 30 s, exit 137 | The main loop breaks, the pipelines close in order: about 0.4 s, exit 0 |
| `--stop-timeout 30` on the container | A slow teardown could be cut short by `SIGKILL` | Docker waits long enough for the clean close |
| Bounded restart policy (`on-failure:3`) | A container that cannot start loops forever under `unless-stopped`. Many looping containers can overload the SOM | It gives up after 3 tries |
| `start` checks health and cleans up | A failed start leaves a container behind | The reason is printed and the container is removed |

## What we measured

Modalix DevKit, one 720p 5 fps stream, YOLOv6s.

| How the container ended | Stop time | Exit code | Buffers still pinned afterwards |
|---|---|---|---|
| `docker stop`, app with the handler | 0.4 s | 0 | none |
| `docker stop -t 30`, app without the handler | 30 s | 137 | none |
| `docker kill` | 2 s | 137 | none |

On this build the kernel releases the decoder, MLA and CMA buffers when the process
exits, however it exits. We could not make an abrupt stop leak memory. The handler is
still the right thing to do: stops are fast, exit codes are meaningful, and the
hardware is shut down in order instead of mid-frame.

If you do see memory that stays allocated after every container is gone, the cause is
elsewhere. Things to check:

- A service outside the containers that holds buffers for them, for example a legacy
  `decoder.service`. It does not exist on a current installation.
- Containers that are still being restarted by Docker: `docker ps -a`.
- The kernel message for the failed allocation: `dmesg | grep -A12 __cma_alloc`. Its
  `range 0:` line lists the free holes, and the last line gives free and total pages.

## Prerequisites

- Build the app with `../app/build.sh` in the Neat SDK container, then copy
  `build/overlay-detector` to `../docker/build/overlay-detector` on the board.
- On the board, in `../docker`: copy `demo.env.example` to `demo.env` and fill in
  `INSIGHT_HOST`, `MEDIA_HOST` and `MODELS_DIR`, then build the empty image once with
  `docker build -t neat-overlay:mounted .`
- Put your YOLOv6s pack in `MODELS_DIR`. It is mounted at `models2/`. The default name
  is `models2/yolov6s_mpk.tar.gz`; set another with `MODEL=...`. The pack must emit
  its box tensors before its class tensors for the in-graph YoloV6 decode.

## Run it (on the board)

```bash
# start and wait until the detector reports "detector running"
./lifecycle.sh start 1 rtsp://<MEDIA_HOST>:8554/mystream_5fps01 0

# graceful stop: prints stop time, exit code and memory
./lifecycle.sh stop 1

# 10 start/stop cycles with a PASS or FAIL verdict
sudo -E ./lifecycle.sh cycle 10
```

Run `cycle` as root to get pinned-buffer accounting from
`/sys/kernel/debug/dma_buf/bufinfo`. That is the reliable leak metric. Without root
only `CmaFree` is shown, and page cache moves that figure by a few MB.

Settings, all overridable from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `models2/yolov6s_mpk.tar.gz` | model pack |
| `FPS` | 5 | source frame rate |
| `STOP_TIMEOUT` | 30 | seconds Docker waits after `SIGTERM` |
| `READY_TIMEOUT` | 60 | seconds to wait for a healthy start |
| `RUN_S` | 45 | seconds each cycle runs |
| `RESTART` | `on-failure:3` | Docker restart policy |
| `PINNED_TOL_MB` | 8 | allowed difference from the pinned baseline |

## What good output looks like

```
>> [start] CmaFree 1713 MB, pinned 1 MB - launching neat-ovc-1 (YOLOv6s, 5 fps)
>> [start] neat-ovc-1 healthy (detector running); CmaFree 1684 MB, pinned 30 MB
>> [stop] CmaFree 1652 MB, pinned 119 MB - docker stop -t 30 neat-ovc-1 (SIGTERM)
>> [stop] graceful close confirmed: exit 0 after 387 ms
>> [stop] CmaFree 1703 MB, pinned 1 MB
```

```
>> [cycle] 3x start/stop, 20s each. Baseline: CmaFree 1707 MB, pinned 1 MB
   cycle  1/3: CmaFree = 1705 MB, pinned = 1 MB
   cycle  2/3: CmaFree = 1712 MB, pinned = 1 MB
   cycle  3/3: CmaFree = 1710 MB, pinned = 1 MB
>> [cycle] PASS: pinned buffers returned to baseline after every stop (no leak).
```

A failed start looks like this, and leaves nothing behind:

```
>> [start] neat-ovc-1 FAILED: state 'running 1' (exited or being restarted) - usually a wrong MODEL path or RTSP URL
[ERR] [io.parse] ModelPack: ... archive path does not exist or is not a regular file: models2/yolov6s_mpk.tar.gz
>> [start] neat-ovc-1 removed
```

## Adapting to your app

- Install a `SIGTERM` / `SIGINT` handler, break your loop, and let the `neat::Run` and
  decoder objects destruct before exit. Do not call `_exit()` or `abort()` on the
  clean-shutdown path.
- Create the container with a `--stop-timeout` longer than your worst-case teardown.
- Use a bounded restart policy while you are still finding out how many streams fit.
