#!/bin/bash
# Combination sweep: K streams per container (independent pipelines inside one overlay-detector process),
# containers added one at a time until a stop rule trips. Finds the max total streams for that K.
#   combo_sweep.sh K [MAX_STREAMS=48]
# env: APP_ARGS (model etc.), STREAM_FMT, PPS_MIN, OUT_DIR, EST_MB (initial RAM estimate per stream), RESERVE_MB
set -uo pipefail
K=${1:?streams per container}; MAXS=${2:-48}
STREAM_FMT=${STREAM_FMT:?printf format of the RTSP URLs, e.g. rtsp://HOST:8554/cam%02d}
APP_ARGS=${APP_ARGS:---fps 5 --decode yolov6 --model models2/yolov6n_mpk.tar.gz}
PPS_MIN=${PPS_MIN:-30}; EST_MB=${EST_MB:-300}; RESERVE_MB=${RESERVE_MB:-250}
OUT=${OUT_DIR:-$PWD/results/combo/k$K}
BOARD_PASS=${BOARD_PASS:?set BOARD_PASS to the board ssh password}
APP_DIR=${APP_DIR:-"~/overlay-demo/docker"}
BOARD=${BOARD:?set BOARD to the SOM IP}
INSIGHT=${INSIGHT:?set INSIGHT to the Neat Insight API base URL, e.g. https://HOST:PORT}
mkdir -p "$OUT/logs"
S=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=8)
bssh() { timeout 90 sshpass -p "$BOARD_PASS" ssh "${S[@]}" sima@"$BOARD" "$@" 2>/dev/null; }
SUM="$OUT/summary.tsv"
printf 'group\tk_this\tstatus\tt_up_s\tstreams\tproducing\tpps_sum\tpps_min\tapp_fps_min\tcpu_idle\tmem_used\tmem_avail\trestarts\tload1\tnote\n' > "$SUM"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/sweep.log"; }

pps_list() { curl -sk -m 15 "$INSIGHT/api/ingest/stats" | python3 -c '
import json,sys
n=int(sys.argv[1]); d=json.load(sys.stdin); m={c["channel"]:c for c in d["channels"]}
for ch in range(n):
    c=m.get(ch); print(ch, c["rtp"].get("packet_rate_pps",0) if c and c.get("active") else 0)' "$1" 2>/dev/null; }
producing_in() {  # count channels lo..hi-1 with pps>=PPS_MIN ; also echoes sum and min over 0..hi-1 as: prod_range prod_all sum min
  pps_list "$2" | python3 -c '
import sys
lo,hi,t=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]); r={}
for l in sys.stdin:
    a=l.split()
    if len(a)==2: r[int(a[0])]=float(a[1])
v=[r.get(c,0) for c in range(hi)]
print(sum(1 for c in range(lo,hi) if r.get(c,0)>=t), sum(1 for x in v if x>=t), round(sum(v)), min(v) if v else 0)' "$1" "$2" "$PPS_MIN"; }
NEWEST=none
snapshot() { bssh 'NEWEST='"$NEWEST"'; idle=$(top -bn1 | awk -F"," "/%Cpu/{for(i=1;i<=NF;i++) if(\$i ~ /id/){gsub(/[^0-9.]/,\"\",\$i); print \$i}}"); rt=0; for c in $(docker ps -a --format "{{.Names}}" | grep -vx "$NEWEST"); do rt=$((rt + $(docker inspect -f "{{.RestartCount}}" $c))); done; echo "${idle:-?} $(free -m | awk "/^Mem:/{print \$3, \$7}") $rt $(cut -d" " -f1 /proc/loadavg)"'; }
fps_min() { bssh 'for c in $(docker ps --format "{{.Names}}" | grep neat-ovc-); do docker logs --tail 40 $c 2>&1 | grep -oE "ch[0-9]+\] frames.* fps=[0-9.]+" | awk "{split(\$1,a,\"]\"); f[a[1]]=\$NF} END{for(k in f) print f[k]}" | cut -d= -f2; done' | sort -n | head -1; }

say "combo sweep start K=$K MAXS=$MAXS args='$APP_ARGS'"
streams=0; g=0; est=$EST_MB; note=""
read -r idle mem avail rt load <<<"$(snapshot)"
while (( streams < MAXS )); do
  ping -c1 -W3 "$BOARD" >/dev/null 2>&1 || { note="board unreachable"; break; }
  fit=$(( (avail - RESERVE_MB) / est )); k=$K; (( k > MAXS - streams )) && k=$((MAXS - streams)); (( fit < k )) && k=$fit
  if (( k < 1 )); then note="stop: mem_avail ${avail} MB cannot fit another stream (est ${est} MB/stream + ${RESERVE_MB} reserve)"; break; fi
  g=$((g+1)); first=$((streams+1)); ch0=$streams
  urls=(); for i in $(seq "$first" $((first+k-1))); do urls+=("$(printf "$STREAM_FMT" "$i")"); done
  extra=""; for u in "${urls[@]:1}"; do extra+=" --url $u"; done
  say "group $g: $k streams (idx $first..$((first+k-1))) est=${est}MB/stream avail=${avail}MB"
  avail_before=$avail; NEWEST=neat-ovc-$g; read -r _i _m _a rt_before _l <<<"$(snapshot)"; t0=$(date +%s)
  bssh "cd $APP_DIR && ./run_overlay.sh $g ${urls[0]} $ch0 $APP_ARGS $extra" >/dev/null || { note="docker run failed for group $g"; break; }
  status=starting; tup=""
  for t in $(seq 1 $((30 + 4*k))); do
    sleep 5
    read -r pr pa psum pmin <<<"$(producing_in "$ch0" "$((ch0+k))")"
    if [[ ${pr:-0} -ge $k ]]; then status=ok; tup=$(( $(date +%s) - t0 )); break; fi
    st=$(bssh "docker inspect -f '{{.State.Status}} {{.RestartCount}}' neat-ovc-$g")
    read -r st_state st_rc <<<"$st"
    if [[ $st_state == exited || ${st_rc:-0} -ge 3 ]]; then status="failed($st)"; tup=$(( $(date +%s) - t0 )); break; fi
  done
  [[ $status == starting ]] && status="no-video(${pr:-0}/$k)"
  bssh "docker logs neat-ovc-$g 2>&1 | grep -vE 'GStreamer-WARNING|Fontconfig' | tail -60" > "$OUT/logs/neat-ovc-$g.log"
  sleep 20
  [[ $status == ok ]] && streams=$((streams+k))
  read -r pr pa psum pmin <<<"$(producing_in 0 "$streams")"
  read -r idle mem avail rt load <<<"$(snapshot)"; fmin=$(fps_min)
  if [[ $status == ok ]]; then d=$(( (avail_before - avail) / k )); (( d > 80 )) && est=$d; fi
  [[ $status == ok && ${pa:-0} -lt $streams ]] && note="older channels dropped: producing ${pa}/${streams}"
  own=$(bssh "docker inspect -f '{{.RestartCount}}' neat-ovc-$g"); [[ ${own:-0} -gt 0 ]] && note="$note start-up relaunches of this group=$own"
  [[ ${rt:-0} -gt ${rt_before:-0} ]] && note="$note OLDER containers restarted ($rt_before->$rt)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$g" "$k" "$status" "${tup:-}" "$streams" "${pa:-?}" "${psum:-?}" "${pmin:-?}" "${fmin:-?}" "$idle" "$mem" "$avail" "$rt" "$load" "$note" | tee -a "$SUM"
  bssh "docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'" > "$OUT/stats-after-g$g.txt"
  if [[ $status != ok ]]; then
    note="group $g ($k streams) $status"; say "$note -> removing it"
    bssh "docker stop -t 5 neat-ovc-$g >/dev/null; sleep 6; docker rm neat-ovc-$g >/dev/null"; sleep 15
    if (( k > 1 )); then K=$((k/2)); say "retrying with smaller group K=$K"; read -r idle mem avail rt load <<<"$(snapshot)"; continue; fi
    break
  fi
  if [[ ${pa:-0} -lt $streams || ${rt:-0} -gt ${rt_before:-0} ]]; then break; fi
  if awk -v i="${idle:-50}" 'BEGIN{exit !(i < 8)}'; then note="stop: cpu idle ${idle}%"; break; fi
done
say "loop done: streams=$streams note='$note'"
NEWEST=none; read -r idle mem avail rt_steady0 load <<<"$(snapshot)"
# steady-state check: 3 samples over 3 minutes
ok=1
for i in 1 2 3; do
  sleep 60
  read -r pr pa psum pmin <<<"$(producing_in 0 "$streams")"; read -r idle mem avail rt load <<<"$(snapshot)"; fmin=$(fps_min)
  printf 'STEADY%s\t\t\t\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' "$i" "$streams" "${pa:-?}" "${psum:-?}" "${pmin:-?}" "${fmin:-?}" "$idle" "$mem" "$avail" "$rt" "$load" | tee -a "$SUM"
  [[ ${pa:-0} -lt $streams || ${rt:-0} -gt ${rt_steady0:-0} ]] && ok=0
done
bssh 'for c in $(docker ps --format "{{.Names}}" | sort -t- -k3 -n); do echo "$c $(docker logs --tail 60 $c 2>&1 | grep -oE "alloc_retries=[0-9]+ dropped=[0-9]+" | sort | uniq -c | tr "\n" " ")"; done; echo $BOARD_PASS | sudo -S -p "" cat /sys/kernel/debug/dma_buf/bufinfo | awk "/exp_name/{next} NF>=5 && \$1 ~ /^[0-9]+\$/ {s[\$5]+=\$1} END{for(k in s) printf \"%s %.0f MB\n\", k, s[k]/1048576}"; free -m; top -bn1 | sed -n 3p; docker stats --no-stream --format "{{.Name}} {{.CPUPerc}} {{.MemUsage}}"' > "$OUT/final-board.txt"
echo "RESULT K=${1} streams=$streams containers=$g steady_ok=$ok cpu_idle=$idle mem_avail=$avail note='$note'" | tee -a "$OUT/sweep.log" | tee -a "$(dirname "$OUT")/RESULTS.txt"
