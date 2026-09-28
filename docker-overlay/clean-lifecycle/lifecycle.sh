#!/bin/bash
# Clean start-up / clean shutdown demo for a YOLOv6s detector container on the SOM.
#
# Shows a start that is verified healthy, a stop that is fast and orderly, and a cycle test
# that proves nothing stays allocated between runs. It relies on two things:
#   1. the app handles SIGTERM and closes its pipelines before exiting (see app/main.cpp)
#   2. `docker stop` gives it time to do so (--stop-timeout), so Docker never has to SIGKILL it.
#
# Usage (run ON the board, next to ../docker/run_overlay.sh with a filled-in demo.env):
#   ./lifecycle.sh start [N] [RTSP_URL] [CH]     # start, wait until healthy, clean up if it fails
#   ./lifecycle.sh stop  [N]                     # graceful stop, report time, exit code, memory
#   ./lifecycle.sh cycle [K] [N] [RTSP_URL] [CH] # K start/stop cycles with a leak verdict
#
# Model: MODEL env (default models2/yolov6s_mpk.tar.gz; put your YOLOv6s pack in MODELS_DIR).
# Run as root (sudo -E ./lifecycle.sh ...) to also get pinned-buffer accounting, the reliable
# leak metric. Without root only CmaFree is shown, which page cache moves by a few MB.
set -uo pipefail
cd "$(dirname "$0")"
OVL=../docker/run_overlay.sh
[[ -x $OVL ]] || { echo "need ../docker/run_overlay.sh (build the app + set demo.env first)"; exit 1; }
[[ -f ../docker/demo.env ]] && . ../docker/demo.env

MODEL=${MODEL:-models2/yolov6s_mpk.tar.gz}   # your YOLOv6s pack (mounted from MODELS_DIR)
FPS=${FPS:-5}
STOP_TIMEOUT=${STOP_TIMEOUT:-30}             # seconds docker waits after SIGTERM before SIGKILL
READY_TIMEOUT=${READY_TIMEOUT:-60}           # seconds to wait for the detector to come up
SETTLE=${SETTLE:-8}                          # seconds to let memory settle after stop
RUN_S=${RUN_S:-45}                           # seconds each cycle runs before it is stopped
RESTART=${RESTART:-on-failure:3}             # bounded: a container that cannot start must not loop forever
PINNED_TOL_MB=${PINNED_TOL_MB:-8}            # pinned buffers may differ from baseline by this much
BUFINFO=/sys/kernel/debug/dma_buf/bufinfo

cma_mb(){ echo $(( $(awk '/CmaFree/{print $2}' /proc/meminfo)/1024 )); }
# Total size of all exported DMA buffers in MB, or "n/a" when the debug file is not readable.
pinned_mb(){
  local out
  if [[ -r $BUFINFO ]]; then out=$(cat $BUFINFO); else out=$(sudo -n cat $BUFINFO 2>/dev/null) || { echo n/a; return; }; fi
  awk '/exp_name/{next} NF>=5 && $1 ~ /^[0-9]+$/ {s+=$1} END{printf "%.0f\n", s/1048576}' <<<"$out"
}
mem(){ local p; p=$(pinned_mb); [[ $p == n/a ]] && echo "CmaFree $(cma_mb) MB" || echo "CmaFree $(cma_mb) MB, pinned ${p} MB"; }
cname(){ echo "neat-ovc-${1:-1}"; }
fail_start(){  # name, reason
  echo ">> [start] $1 FAILED: $2"
  docker logs "$1" 2>&1 | grep -vE "GStreamer-WARNING|Fontconfig|^\s*$" | tail -5
  docker rm -f "$1" >/dev/null 2>&1      # never leave a failing container behind to crash-loop
  echo ">> [start] $1 removed"
  return 1
}

start(){
  local N=${1:-1} URL=${2:-rtsp://${MEDIA_HOST:?set MEDIA_HOST in demo.env}:8554/mystream_5fps01} CH=${3:-0}
  local n; n=$(cname "$N")
  if docker inspect "$n" >/dev/null 2>&1; then echo ">> [start] $n already exists; stop it first"; return 1; fi
  echo ">> [start] $(mem) - launching $n (YOLOv6s, ${FPS} fps)"
  # The docker args after "--" override run_overlay.sh's defaults: a stop timeout long enough for a
  # clean close, and a bounded restart policy.
  "$OVL" "$N" "$URL" "$CH" --fps "$FPS" --decode yolov6 --model "$MODEL" -- \
      --stop-timeout "$STOP_TIMEOUT" --restart "$RESTART" >/dev/null || { echo ">> [start] run_overlay.sh failed"; docker rm -f "$n" >/dev/null 2>&1; return 1; }
  local st
  for _ in $(seq 1 "$READY_TIMEOUT"); do
    st=$(docker inspect -f '{{.State.Status}} {{.RestartCount}}' "$n" 2>/dev/null) || { fail_start "$n" "container disappeared"; return 1; }
    case $st in
      "running 0") docker logs "$n" 2>&1 | grep -q "detector running" && { echo ">> [start] $n healthy (detector running); $(mem)"; return 0; } ;;
      *) fail_start "$n" "state '$st' (exited or being restarted) - usually a wrong MODEL path or RTSP URL"; return 1 ;;
    esac
    sleep 1
  done
  fail_start "$n" "not ready within ${READY_TIMEOUT}s"; return 1
}

stop(){
  local N=${1:-1} n; n=$(cname "$N")
  docker inspect "$n" >/dev/null 2>&1 || { echo ">> [stop] $n does not exist"; return 1; }
  echo ">> [stop] $(mem) - docker stop -t ${STOP_TIMEOUT} $n (SIGTERM)"
  local t0 t1 ec
  t0=$(date +%s%N); docker stop -t "$STOP_TIMEOUT" "$n" >/dev/null 2>&1; t1=$(date +%s%N)
  ec=$(docker inspect -f '{{.State.ExitCode}}' "$n" 2>/dev/null)
  if [[ $ec == 0 ]] && docker logs "$n" 2>&1 | grep -q "clean shutdown: releasing run"; then
    echo ">> [stop] graceful close confirmed: exit 0 after $(( (t1-t0)/1000000 )) ms"
  else
    echo ">> [stop] WARNING: not a graceful close (exit ${ec:-?} after $(( (t1-t0)/1000000 )) ms)."
    echo "          Exit 137 after ~${STOP_TIMEOUT}s means the app ignored SIGTERM and Docker killed it."
  fi
  docker rm "$n" >/dev/null 2>&1
  sleep "$SETTLE"
  echo ">> [stop] $(mem)"
}

cycle(){
  local K=${1:-10} N=${2:-1} URL=${3:-} CH=${4:-0}
  local base_c base_p p ok=1; base_c=$(cma_mb); base_p=$(pinned_mb)
  echo ">> [cycle] ${K}x start/stop, ${RUN_S}s each. Baseline: $(mem)"
  for i in $(seq 1 "$K"); do
    start "$N" "$URL" "$CH" >/dev/null 2>&1 || { echo "   cycle $i: start failed (run './lifecycle.sh start' to see why)"; ok=0; break; }
    sleep "$RUN_S"
    stop "$N" >/dev/null 2>&1
    p=$(pinned_mb)
    if [[ $p == n/a ]]; then printf "   cycle %2d/%s: CmaFree = %s MB\n" "$i" "$K" "$(cma_mb)"; else printf "   cycle %2d/%s: CmaFree = %s MB, pinned = %s MB\n" "$i" "$K" "$(cma_mb)" "$p"; fi
    [[ $p != n/a && $base_p != n/a ]] && (( p > base_p + PINNED_TOL_MB )) && ok=0
  done
  if [[ $base_p == n/a ]]; then
    echo ">> [cycle] done. Pinned-buffer accounting needs root (sudo -E ./lifecycle.sh cycle ...)."
    echo "           CmaFree alone moves a few MB with page cache; only a steady downward trend matters."
  elif [[ $ok == 1 ]]; then echo ">> [cycle] PASS: pinned buffers returned to baseline after every stop (no leak)."
  else echo ">> [cycle] FAIL: pinned buffers stayed above baseline or a start failed."; return 1
  fi
}

cmd=${1:-}; shift || true
case "$cmd" in
  start) start "$@";;
  stop)  stop "$@";;
  cycle) cycle "$@";;
  *) sed -n '2,18p' "$0"; exit 2;;
esac
