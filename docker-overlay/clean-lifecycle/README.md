# Clean start-up / clean shutdown for a YOLOv6s container

A sample that shows how to start and **stop** a YOLOv6s detector container on the
SOM so that every stop **fully releases the MLA, the HW decoder and their
contiguous CMA** — and repeated start/stop cycles do not leak. This is the fix for
the *"resources aren't reclaimed between runs / `EBUSY` while `CmaFree` is high /
only a reboot recovers it"* symptom.

## Why memory leaks across runs (and how this fixes it)

The decoder and the model Run hold **contiguous CMA** (order-9/10 blocks). They are
released in the neat objects' **destructors**. If the process is terminated
*abruptly*, destructors never run, so those handles and their CMA are never freed —
and because CMA can't be compacted, the next big allocation `EBUSY`s even though
`CmaFree` looks high, until a reboot.

Two things cause an abrupt termination, and this sample fixes both:

1. **No SIGTERM handler in the app.** `docker stop` sends `SIGTERM`; with no
   handler the default action kills the process with no stack unwinding → no
   destructors. **Fix:** `app/main.cpp` now installs a `SIGTERM`/`SIGINT` handler
   that sets a stop flag; the main loop breaks, `run_stream()` returns normally,
   and the `neat::Run` / `neat::Graph` objects destruct → MLA + decoder + CMA
   released. (It also no longer `_exit()`s the multi-stream path on a clean stop,
   which would skip destructors too.)

2. **`docker stop` not giving it time.** After `SIGTERM`, Docker waits only
   `--stop-timeout` (default 10 s) before `SIGKILL`. If teardown needs longer it
   gets killed → same leak. **Fix:** the container is created with
   `--stop-timeout 30`, and `docker stop -t 30` is used.

Corollary: **never** tear a detector down with `kill -9` / `docker kill` / a short
`docker stop -t 1`. That is the leak.

## Prerequisites

- Build the app: `../app/build.sh` in the Neat SDK container, then copy
  `build/overlay-detector` to `../docker/build/overlay-detector` on the board.
- `../docker/demo.env` filled in (`INSIGHT_HOST`, `MEDIA_HOST`, `MODELS_DIR`).
- Your **YOLOv6s** pack in `MODELS_DIR` (default mount → `models2/`), e.g.
  `models2/yolov6s_mpk.tar.gz`. Set a different name with `MODEL=...`.

## Run it (on the board)

```bash
# one clean start (waits until the detector reports "detector running")
./lifecycle.sh start 1 rtsp://<MEDIA_HOST>:8554/mystream_5fps01 0

# clean stop — prints CmaFree before/after and confirms a graceful teardown
./lifecycle.sh stop 1

# the real proof: K start/stop cycles, CmaFree printed after each
./lifecycle.sh cycle 10
```

`MODEL`, `FPS`, `STOP_TIMEOUT`, `READY_TIMEOUT` are env-overridable.

## What good output looks like

```
>> [start] CmaFree baseline = 1802 MB — launching neat-ovc-1 (YOLOv6s, 5 fps)
>> [start] neat-ovc-1 healthy (detector running); CmaFree now 1520 MB
>> [stop]  CmaFree before = 1520 MB — docker stop -t 30 neat-ovc-1 (SIGTERM)
>> [stop]  graceful teardown confirmed (app closed the Run/decoder)
>> [stop]  CmaFree after = 1800 MB  (recovered 280 MB)
```

And across cycles, `CmaFree` returns to ~baseline every time:

```
   cycle  1/10: CmaFree = 1800 MB
   cycle  5/10: CmaFree = 1801 MB
   cycle 10/10: CmaFree = 1800 MB
```

**Flat across cycles = clean reclamation.** A steady downward trend, or a missing
"graceful teardown confirmed" line, means the process was killed before it could
close — check that nothing is `kill -9`ing it and that `--stop-timeout` is long
enough.

## Adapting to your app

The same two rules apply to any detector, not just this overlay example:
- Install a `SIGTERM`/`SIGINT` handler → break your loop → let the `neat::Run` /
  decoder objects destruct (or call their `close()` explicitly) before exit. Never
  `_exit()`/`abort()` on the clean-shutdown path.
- Create the container with a `--stop-timeout` longer than your worst-case
  teardown, and stop it with `docker stop` (SIGTERM), never `docker kill`.
