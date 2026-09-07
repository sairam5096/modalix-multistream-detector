# Runtime model switching

Switching the model on a camera does **not** load a model — it re-wires the
graph. All models are compiled and resident on the MLA from startup.

An interactive block diagram + timeline is in
[`model-switching.html`](model-switching.html). The sequence:

| # | Step | What happens |
|---|------|--------------|
| 1 | **Select** | Pick one or more cameras + a model in the control panel (`:8092/control`). |
| 2 | **Request** | The browser sends the change; the host proxies it to the SoM: `POST /api/streams/{id}/models {"models":["coco-m"]}` → `:8600`. |
| 3 | **Coalesce** | The SoM updates that camera's model set and stamps a rebuild request. A worker debounces **~0.6 s**, so a burst of clicks becomes a single rebuild. |
| 4 | **Re-wire** | The fused graph is rebuilt — each camera's decoder is reconnected to its chosen shared detector. Models are already warm, so **nothing loads from disk**. Only models still in use are built. |
| 5 | **Resume** | The new graph goes live (`generation++`) and the pump reads its outputs. Cameras blink for **~2 s**, then run at full rate. |

## Why it's fast

- **Warm registry** — models are loaded once at boot; a switch is a graph
  rewire, not a model load.
- **Shared detectors** — one stage per model, not per camera; switching a camera
  just moves its fan-in link to a different (already built) stage, or triggers
  building a stage for a newly-used model.
- **Coalesced rebuild** — rapid multi-camera changes collapse into one rebuild
  instead of a storm.

## Examples

```bash
# one camera to the accurate model
curl -X POST http://<SOM_IP>:8600/api/streams/0/models \
     -H 'Content-Type: application/json' -d '{"models":["coco-m"]}'

# every camera to the fast model
curl -X POST http://<SOM_IP>:8600/api/models_all \
     -H 'Content-Type: application/json' -d '{"models":["coco-n"]}'

# chain two models on one camera (boxes merged)
curl -X POST http://<SOM_IP>:8600/api/streams/3/models \
     -H 'Content-Type: application/json' -d '{"models":["coco-n","coco-v8"]}'
```

## Caveats

- Every rebuild briefly re-glitches **all** cameras (~2 s). Good for occasional
  switches; not for rapid flipping — a fast burst of rebuilds can degrade the
  decoder daemon (a detector relaunch, or a reboot, clears it).
- Crop enable/disable is **host-side** and instant — it never rebuilds the SoM
  graph.
- The accurate BF16 model fits ~6 cameras on one MLA; beyond that the MLA is
  throughput-bound and the graph can stall.
