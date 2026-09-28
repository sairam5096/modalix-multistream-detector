#!/bin/bash
# Clean start-up / clean shutdown demo for a YOLOv6s detector container on the SOM.
#
# The point: prove that stopping the container RELEASES the MLA, the HW decoder and
# their contiguous CMA every time — so repeated start/stop cycles do NOT leak CMA
# (the "resources aren't reclaimed between runs / EBUSY at high CmaFree / needs a
# reboot" symptom). It relies on two things working together:
#   1. the app handles SIGTERM and tears the Run down via destructors (see app/main.cpp)
#   2. `docker stop` gives it time to do so (--stop-timeout below), i.e. no SIGKILL.
#
# Usage (run ON the board, next to run_overlay.sh with a filled-in demo.env):
#   ./lifecycle.sh start [N] [RTSP_URL] [CH]     # clean start, wait until healthy
#   ./lifecycle.sh stop  [N]                     # clean stop, verify CMA reclaimed
#   ./lifecycle.sh cycle [K] [N] [RTSP_URL] [CH] # K start/stop cycles, print CmaFree each time
#
# Model: MODEL env (default models2/yolov6s_mpk.tar.gz — put your YOLOv6s pack in MODELS_DIR).
set -uo pipefail
cd "$(dirname "$0")"
OVL=../docker/run_overlay.sh
[[ -x $OVL ]] || { echo "need ../docker/run_overlay.sh (build the app + set demo.env first)"; exit 1; }
[[ -f ../docker/demo.env ]] && . ../docker/demo.env

MODEL=${MODEL:-models2/yolov6s_mpk.tar.gz}   # your YOLOv6s pack (mounted from MODELS_DIR)
FPS=${FPS:-5}
STOP_TIMEOUT=${STOP_TIMEOUT:-30}             # seconds docker waits after SIGTERM before SIGKILL
READY_TIMEOUT=${READY_TIMEOUT:-60}           # seconds to wait for the detector to come up
SETTLE=${SETTLE:-8}                          # seconds to let CMA settle after stop

cma_kb(){ awk '/CmaFree/{print $2}' /proc/meminfo; }
cma_mb(){ echo $(( $(cma_kb)/1024 )); }
cname(){ echo "neat-ovc-${1:-1}"; }

start(){
  local N=${1:-1} URL=${2:-rtsp://${MEDIA_HOST:?set MEDIA_HOST in demo.env}:8554/mystream_5fps01} CH=${3:-0}
  local base; base=$(cma_mb)
  echo ">> [start] CmaFree baseline = ${base} MB — launching $(cname "$N") (YOLOv6s, ${FPS} fps)"
  # --stop-timeout on the container so `docker stop` honours SIGTERM long enough to close cleanly.
  "$OVL" "$N" "$URL" "$CH" --fps "$FPS" --decode yolov6 --model "$MODEL" -- \
      --stop-timeout "$STOP_TIMEOUT" >/dev/null || { echo "run_overlay.sh failed"; return 1; }
  # Clean start-up gate: wait until the detector actually reports it is running.
  local n; n=$(cname "$N")
  for _ in $(seq 1 "$READY_TIMEOUT"); do
    if docker logs "$n" 2>&1 | grep -q "detector running"; then
      echo ">> [start] $n healthy (detector running); CmaFree now $(cma_mb) MB"
      return 0
    fi
    docker inspect -f '{{.State.Running}}' "$n" 2>/dev/null | grep -qx true || { echo ">> [start] $n exited early:"; docker logs "$n" 2>&1 | tail -5; return 1; }
    sleep 1
  done
  echo ">> [start] $n did not report ready in ${READY_TIMEOUT}s"; docker logs "$n" 2>&1 | tail -5; return 1
}

stop(){
  local N=${1:-1} n; n=$(cname "$N")
  local before; before=$(cma_mb)
  echo ">> [stop] CmaFree before = ${before} MB — docker stop -t ${STOP_TIMEOUT} $n (SIGTERM)"
  docker stop -t "$STOP_TIMEOUT" "$n" >/dev/null 2>&1
  # Confirm it was a GRACEFUL close, not a SIGKILL: the app logs this on the SIGTERM path.
  if docker logs "$n" 2>&1 | grep -q "clean shutdown: releasing run"; then
    echo ">> [stop] graceful teardown confirmed (app closed the Run/decoder)"
  else
    echo ">> [stop] WARNING: no clean-shutdown log — the app may have been SIGKILLed (CMA will leak)."
  fi
  docker rm "$n" >/dev/null 2>&1
  sleep "$SETTLE"
  local after; after=$(cma_mb)
  echo ">> [stop] CmaFree after = ${after} MB  (recovered $(( after - before )) MB)"
}

cycle(){
  local K=${1:-10} N=${2:-1} URL=${3:-} CH=${4:-0}
  local base; base=$(cma_mb)
  echo ">> [cycle] ${K}x start/stop — CmaFree should return to ~${base} MB every time (no leak)"
  for i in $(seq 1 "$K"); do
    start "$N" "$URL" "$CH" >/dev/null 2>&1 || { echo "cycle $i: start failed"; break; }
    sleep 45   # run a while
    stop "$N" >/dev/null 2>&1
    printf "   cycle %2d/%s: CmaFree = %s MB\n" "$i" "$K" "$(cma_mb)"
  done
  echo ">> [cycle] done. Flat CmaFree across cycles = clean reclamation; a downward trend = a leak."
}

cmd=${1:-}; shift || true
case "$cmd" in
  start) start "$@";;
  stop)  stop "$@";;
  cycle) cycle "$@";;
  *) sed -n '2,20p' "$0"; exit 2;;
esac
