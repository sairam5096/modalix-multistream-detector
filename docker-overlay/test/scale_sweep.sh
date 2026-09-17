#!/bin/bash
# Scaling sweep for on-SOM-overlay containers (neat-ovc-N: stream NN -> Insight video channel N-1).
# Health = Insight video RTP rate on the container's channel (overlay video only, no metadata).
# Usage: scale_sweep.sh [MAX=48] [GAP_S=15] [START=1]
set -uo pipefail
MAX=${1:-48}; GAP=${2:-15}; START=${3:-1}
STREAM_FMT=${STREAM_FMT:?printf format of the RTSP URLs, e.g. rtsp://HOST:8554/cam%02d}
SRC_FPS=${SRC_FPS:-5}
PPS_MIN=${PPS_MIN:-30}
OUT=${OUT_DIR:-$PWD/results/sweep}
BOARD_PASS=${BOARD_PASS:?set BOARD_PASS to the board ssh password}
APP_DIR=${APP_DIR:-"~/overlay-demo/docker"}
RUNNER=${RUNNER:-./run_overlay.sh}
NAME=${NAME:-neat-ovc}                 # container name prefix used by run_overlay.sh
APP_ARGS=${APP_ARGS:---fps $SRC_FPS}   # extra app args
CMA_MIN=${CMA_MIN:-120}                # stop adding containers below this CmaFree (MB)
BOARD=${BOARD:?set BOARD to the SOM IP}
INSIGHT=${INSIGHT:?set INSIGHT to the Neat Insight API base URL, e.g. https://HOST:PORT}
mkdir -p "$OUT/logs"
S=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=8)
bssh() { timeout 60 sshpass -p "$BOARD_PASS" ssh "${S[@]}" sima@"$BOARD" "$@" 2>/dev/null; }
SUM="$OUT/summary.tsv"
[[ -f $SUM && $START -gt 1 ]] || printf 'N\tstatus\tt_up_s\trunning\tproducing\tvideo_pps_sum\tmin_pps\tmax_pps\tapp_fps_min\tcpu_idle_pct\tmem_used_mb\tcma_free_mb\tload1\tnote\n' > "$SUM"
running=$((START-1)); fail_note=""

chan_pps() {  # "ch pps" for channels 0..running-1
  curl -sk -m 15 "$INSIGHT/api/ingest/stats" | python3 -c '
import json,sys
n=int(sys.argv[1]); d=json.load(sys.stdin); m={c["channel"]:c for c in d["channels"]}
for ch in range(n):
    c=m.get(ch); print(ch, c["rtp"].get("packet_rate_pps",0) if c and c.get("active") else 0)' "$running" 2>/dev/null
}
board_snapshot() {  # cpu_idle mem cma load
  bssh 'idle=$(top -bn1 | awk -F"," "/%Cpu/{for(i=1;i<=NF;i++) if(\$i ~ /id/){gsub(/[^0-9.]/,\"\",\$i); print \$i}}"); echo "${idle:-?} $(free -m | awk "/^Mem:/{print \$3}") $(( $(grep CmaFree /proc/meminfo | awk "{print \$2}") / 1024 )) $(cut -d" " -f1 /proc/loadavg)"'
}
app_fps_min() {  # min of the latest app-reported fps across running containers
  bssh 'for c in $(docker ps --format "{{.Names}}" | grep "$NAME"-); do docker logs --tail 3 $c 2>&1 | grep -oE "fps=[0-9.]+" | tail -1 | cut -d= -f2; done' | sort -n | head -1
}

echo "overlay sweep start $(date -u +%FT%TZ) MAX=$MAX GAP=${GAP}s START=$START streams=$STREAM_FMT fps=$SRC_FPS pps_min=$PPS_MIN" | tee -a "$OUT/sweep.log"
for N in $(seq "$START" "$MAX"); do
  ping -c1 -W3 "$BOARD" >/dev/null 2>&1 || { fail_note="board unreachable before $N"; echo "$fail_note" | tee -a "$OUT/sweep.log"; break; }
  t0=$(date +%s)
  url=$(printf "$STREAM_FMT" "$N")
  bssh "cd $APP_DIR && $RUNNER $N $url $((N-1)) $APP_ARGS" >/dev/null || { fail_note="docker run failed for $N"; echo "$fail_note" | tee -a "$OUT/sweep.log"; break; }
  status=starting; tup=""
  for t in $(seq 1 30); do
    sleep 5
    state=$(bssh "docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' $NAME-$N")
    pps=$(curl -sk -m 15 "$INSIGHT/api/ingest/stats" | python3 -c 'import json,sys; ch=int(sys.argv[1]); d=json.load(sys.stdin); c=[x for x in d["channels"] if x["channel"]==ch and x.get("active")]; print(c[0]["rtp"].get("packet_rate_pps",0) if c else 0)' "$((N-1))" 2>/dev/null || echo 0)
    if awk -v r="$pps" -v t="$PPS_MIN" 'BEGIN{exit !(r>=t)}'; then status=ok; tup=$(( $(date +%s) - t0 )); break; fi
    if [[ $state != running* ]]; then status=failed; tup=$(( $(date +%s) - t0 )); break; fi
  done
  [[ $status == starting ]] && status=no-video-150s
  bssh "docker logs $NAME-$N 2>&1 | tail -40" > "$OUT/logs/$NAME-$N.log"
  if [[ $status == ok ]]; then running=$N; else fail_note="container $N: $status ($(bssh "docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' $NAME-$N"))"; fi
  sleep "$GAP"
  producing=0; sum=0; mn=99999; mx=0
  while read -r ch r; do [[ -z $ch ]] && continue; awk -v r="$r" -v t="$PPS_MIN" 'BEGIN{exit !(r>=t)}' && producing=$((producing+1)); sum=$(python3 -c "print(round($sum+$r))"); mn=$(python3 -c "print(min($mn,$r))"); mx=$(python3 -c "print(max($mx,$r))"); done <<<"$(chan_pps)"
  read -r idle mem cma load <<<"$(board_snapshot)"
  fmin=$(app_fps_min)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$N" "$status" "${tup:-}" "$running" "$producing" "$sum" "$mn" "$mx" "${fmin:-?}" "$idle" "$mem" "$cma" "$load" "$fail_note" | tee -a "$SUM"
  bssh "docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'" > "$OUT/stats-after-$N.txt"
  [[ $status != ok ]] && break
  # stop when the CPU is nearly saturated or memory is nearly gone: the next start would only degrade all
  awk -v i="${idle:-50}" -v m="${mem:-0}" -v c="${cma:-9999}" -v cm="$CMA_MIN" 'BEGIN{exit !(i < 8 || m > 5300 || c < cm)}' && { fail_note="stopping: cpu idle ${idle}% mem ${mem} MB cma ${cma} MB"; echo "$fail_note" | tee -a "$OUT/sweep.log"; break; }
done
echo "overlay sweep loop done $(date -u +%FT%TZ): running=$running note='${fail_note}'" | tee -a "$OUT/sweep.log"
sleep 120
producing=0; sum=0; mn=99999; mx=0
while read -r ch r; do [[ -z $ch ]] && continue; awk -v r="$r" -v t="$PPS_MIN" 'BEGIN{exit !(r>=t)}' && producing=$((producing+1)); sum=$(python3 -c "print(round($sum+$r))"); mn=$(python3 -c "print(min($mn,$r))"); mx=$(python3 -c "print(max($mx,$r))"); done <<<"$(chan_pps)"
read -r idle mem cma load <<<"$(board_snapshot)"; fmin=$(app_fps_min)
printf 'STEADY\t2min\t\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tsteady-state\n' "$running" "$producing" "$sum" "$mn" "$mx" "${fmin:-?}" "$idle" "$mem" "$cma" "$load" | tee -a "$SUM"
curl -sk -m 20 "$INSIGHT/api/ingest/stats" -o "$OUT/ingest-final.json"
bssh "docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'; free -m; grep -E 'CmaTotal|CmaFree' /proc/meminfo; uptime; top -bn1 | sed -n 3p" > "$OUT/final-board.txt"
echo "OVERLAY SWEEP COMPLETE running=$running" | tee -a "$OUT/sweep.log"
