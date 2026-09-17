#!/bin/bash
# Unattended endurance + monkey harness for the on-SOM overlay containers (neat-ovc-N, C++).
# Runs independently of any interactive session (start with: CHAOS=1 setsid nohup endurance.sh run &).
# Health = Insight video RTP packet rate per channel (the overlay containers send encoded video, no metadata).
# Every sample also appends all container logs (docker logs -t --since last), new dmesg lines and, every 10th
# sample, docker stats + dma-buf accounting into $OUT_DIR/logs/.
#
#   endurance.sh run            main loop: sample every 60 s, chaos every CHAOS_EVERY_MIN, self-heal hung containers
#   endurance.sh report         summarize samples/alerts/restarts so far
#   endurance.sh stop           ask the loop to stop after the current iteration
#
# Layout under test is whatever is running on the board (containers named neat-ovc-N,
# container N -> Insight channel N-1). Expected rate per channel is learned from the first 10 samples.
set -uo pipefail
BOARD=${BOARD:?set BOARD to the SOM IP}
INSIGHT=${INSIGHT:?set INSIGHT to the Neat Insight API base URL, e.g. https://HOST:PORT}
OUT=${OUT_DIR:-$PWD/results/endurance}
BOARD_PASS=${BOARD_PASS:?set BOARD_PASS to the board ssh password}
APP_DIR=${APP_DIR:-"~/overlay-demo/docker"}
CHAOS_EVERY_MIN=${CHAOS_EVERY_MIN:-20}
CHAOS=${CHAOS:-0}
mkdir -p "$OUT"
SAMPLES="$OUT/samples.tsv"; ALERTS="$OUT/alerts.log"; EVENTS="$OUT/events.log"; STOPF="$OUT/STOP"; EXP="$OUT/expected_rates.json"
S=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=8)
bssh() { timeout 90 sshpass -p "$BOARD_PASS" ssh "${S[@]}" sima@"$BOARD" "$@" 2>/dev/null; }
now() { date -u +%FT%TZ; }
alert() { echo "$(now) $*" | tee -a "$ALERTS"; }
event() { echo "$(now) $*" >> "$EVENTS"; }

[[ -f $SAMPLES ]] || printf 'time\tup\texited\trestarting\trestarts_total\tchannels\trate_sum\trate_min\tlow\tmem_used_mb\tcache_mb\tcma_free_mb\tload1\ttemp_c\tkerr\tdocker_ok\n' > "$SAMPLES"

LOGS="$OUT/logs"; mkdir -p "$LOGS"
# ---------- log capture (all container stdout/stderr since the last pull, new dmesg lines, periodic stats) ----------
collect_logs() {
  bssh 'ts=$(cat /tmp/endur_ts 2>/dev/null || echo 10m); n=$(date -u +%Y-%m-%dT%H:%M:%SZ);
        for c in $(docker ps -a --format "{{.Names}}" | grep "^neat-ovc-" | sort -t- -k3 -n); do docker logs -t --since "$ts" --until "$n" "$c" 2>&1 | sed "s/^/$c /"; done; echo "$n" > /tmp/endur_ts' >> "$LOGS/containers.log"
  bssh 'l=$(cat /tmp/endur_dmesg_n 2>/dev/null || echo 0); dmesg 2>/dev/null | tail -n +$((l+1)); dmesg 2>/dev/null | wc -l > /tmp/endur_dmesg_n' | sed "s/^/$(now) /" >> "$LOGS/dmesg.log"
  if (( SAMPLE_N % 10 == 1 )); then
    { echo "===== $(now)"; bssh 'docker stats --no-stream --format "{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}} pids={{.PIDs}}" | sort -t- -k3 -n; docker ps -a --format "{{.Names}} {{.Status}} restarts=" | while read -r n st r; do echo "$n $st restarts=$(docker inspect -f "{{.RestartCount}}" $n) args=[$(docker inspect -f "{{join .Args \" \"}}" $n)]"; done'; } >> "$LOGS/docker_stats.log"
    { echo "===== $(now)"; bssh 'echo $BOARD_PASS | sudo -S -p "" cat /sys/kernel/debug/dma_buf/bufinfo 2>/dev/null | awk "/exp_name/{next} NF>=5 && \$1 ~ /^[0-9]+$/ {s[\$5]+=\$1; n[\$5]++} END{for(k in s) printf \"%s %d bufs %.0f MB\\n\", k, n[k], s[k]/1048576}"; free -m | sed -n 2p; grep -E "CmaFree|CmaTotal" /proc/meminfo | tr "\n" " "; echo; cat /sys/class/hwmon/hwmon1/temp1_input 2>/dev/null'; } >> "$LOGS/board_mem.log"
  fi
}

# ---------- one health sample ----------
sample() {
  local b
  b=$(bssh 'up=$(docker ps -q | wc -l); ex=$(docker ps -aq --filter status=exited | wc -l); rs=$(docker ps -aq --filter status=restarting | wc -l);
      rt=0; for c in $(docker ps -aq); do rt=$((rt + $(docker inspect -f "{{.RestartCount}}" $c))); done;
      mem=$(free -m | awk "/^Mem:/{print \$3}"); cache=$(free -m | awk "/^Mem:/{print \$6}"); cma=$(( $(grep CmaFree /proc/meminfo | awk "{print \$2}") / 1024 ));
      load=$(cut -d" " -f1 /proc/loadavg); t=$(cat /sys/class/hwmon/hwmon1/temp1_input 2>/dev/null); t=$(( ${t:-0} / 1000 ));
      kerr=$(dmesg 2>/dev/null | grep -ciE "oom|killed process|mla.*(error|fault|timeout)|cvu.*(error|fault)|allegro.*error|hung task|soft lockup");
      dok=$(docker info >/dev/null 2>&1 && echo 1 || echo 0);
      echo "$up $ex $rs $rt $mem $cache $cma $load $t $kerr $dok"')
  if [[ -z $b ]]; then
    printf '%s\tBOARD_UNREACHABLE\n' "$(now)" >> "$SAMPLES"; alert "BOARD UNREACHABLE (ssh)"; return 1
  fi
  read -r up ex rs rt mem cache cma load temp kerr dok <<<"$b"
  local ins
  ins=$(curl -sk -m 20 "$INSIGHT/api/ingest/stats" | python3 -c '
import json,sys,os
d=json.load(sys.stdin); cs={c["channel"]:c["rtp"].get("packet_rate_pps",0) for c in d["channels"] if c.get("active")}
exp={}
try: exp={int(k):v for k,v in json.load(open(sys.argv[1])).items()}
except Exception: pass
low=[f"ch{k}:{cs.get(k,0)}" for k in sorted(exp) if cs.get(k,0) < 0.3*exp[k]]
vals=list(cs.values()) or [0]
print(len(cs), round(sum(vals),1), min(vals), ",".join(low) or "-")
json.dump(cs, open(sys.argv[2],"w"))' "$EXP" "$OUT/last_rates.json" 2>/dev/null || echo "? ? ? ?")
  read -r ch rsum rmin low <<<"$ins"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(now)" "$up" "$ex" "$rs" "$rt" "$ch" "$rsum" "$rmin" "$low" "$mem" "$cache" "$cma" "$load" "$temp" "$kerr" "$dok" >> "$SAMPLES"
  python3 -c 'import json,sys; d={int(k):v for k,v in json.load(open(sys.argv[1])).items()}; print(sys.argv[2]+"\t"+"\t".join(f"{k}:{d[k]:.0f}" for k in sorted(d)))' "$OUT/last_rates.json" "$(now)" >> "$LOGS/rates.tsv" 2>/dev/null
  collect_logs
  # learn expected rates from the first 10 samples (median-ish: just take max seen)
  python3 - "$EXP" "$OUT/last_rates.json" <<'EOF'
import json,sys,os
exp_p, last_p = sys.argv[1], sys.argv[2]
try: last={int(k):v for k,v in json.load(open(last_p)).items()}
except Exception: sys.exit(0)
exp={}
try: exp={int(k):v for k,v in json.load(open(exp_p)).items()}
except Exception: pass
n=int(os.environ.get("SAMPLE_N","0"))
if n <= 10:
    for k,v in last.items(): exp[k]=max(exp.get(k,0), v)
    json.dump(exp, open(exp_p,"w"))
EOF
  # alert conditions
  [[ $dok != 1 ]] && alert "docker daemon not responding"
  [[ ${kerr:-0} -gt ${KERR_BASE:-2} ]] && alert "kernel error count rose to $kerr (baseline ${KERR_BASE:-2})"
  [[ ${rt:-0} -gt ${LAST_RT:-0} ]] && alert "container restart count rose: $LAST_RT -> $rt" && bssh 'for c in $(docker ps -a --format "{{.Names}}"); do r=$(docker inspect -f "{{.RestartCount}}" $c); [ "$r" -gt 0 ] && echo "   $c restarts=$r"; done' | tee -a "$ALERTS"
  LAST_RT=$rt
  [[ ${ex:-0} -gt 0 ]] && alert "exited containers: $(bssh 'docker ps -a --filter status=exited --format "{{.Names}}({{.Status}})" | tr "\n" " "')"
  if [[ $low != "-" && $low != "?" ]]; then LOW_STREAK=$((LOW_STREAK+1)); else LOW_STREAK=0; fi
  [[ $LOW_STREAK -ge 3 ]] && alert "channels below 30% of expected video rate for 3 samples: $low"
  [[ ${cma:-999} -lt 120 && ${cache:-0} -lt 300 ]] && alert "CMA free ${cma} MB with little cache to reclaim"
  [[ ${mem:-0} -gt 5300 ]] && alert "RAM used ${mem} MB (OOM risk)"
  [[ ${temp:-0} -gt 85 ]] && alert "SoC temperature ${temp} C"
  # self-heal: a container that is 'running' but whose channel has been silent for 3 samples
  python3 - "$OUT/last_rates.json" "$EXP" "$OUT/silent.json" <<'EOF' | while read -r n; do event "WATCHDOG: channel silent 3x -> docker restart neat-ovc-$n"; alert "watchdog restart neat-ovc-$n"; bssh "docker restart -t 5 neat-ovc-$n 2>/dev/null"; done
import json,sys
last={int(k):v for k,v in json.load(open(sys.argv[1])).items()} if True else {}
try: exp={int(k):v for k,v in json.load(open(sys.argv[2])).items()}
except Exception: exp={}
try: silent=json.load(open(sys.argv[3]))
except Exception: silent={}
for ch in exp:
    key=str(ch)
    silent[key] = silent.get(key,0)+1 if last.get(ch,0) < 20 else 0
    if silent[key] >= 3:
        print(ch+1); silent[key]=0
json.dump(silent, open(sys.argv[3],"w"))
EOF
  return 0
}

# ---------- chaos / monkey actions (rotating; one container at a time, MLA-friendly pacing) ----------
read -r -a VARIANTS <<<"${VARIANTS_LIST:-default yolov6n fps10 res360 default yolov6n}"
capture_before() { bssh "mkdir -p /data/endurance_logs; { echo \"===== \$(date -u +%FT%TZ) $1 before chaos: $2\"; docker logs -t --tail 200 $1 2>&1; } >> /data/endurance_logs/$1.log"; }
chaos() {
  local i=$1
  local names n
  names=($(bssh 'docker ps --format "{{.Names}}" | grep -E "^neat-ovc-" | sort -t- -k3 -n'))
  [[ ${#names[@]} -eq 0 ]] && return
  n=${names[$((RANDOM % ${#names[@]}))]}
  local idx=${n##*-}
  case $((i % 6)) in
    0) event "CHAOS graceful restart $n"; capture_before "$n" "graceful restart"; bssh "docker restart -t 5 $n" >/dev/null ;;
    1) event "CHAOS process crash (kill -9 app pid) $n -> expect docker auto-relaunch"; capture_before "$n" "kill -9";
       bssh "p=\$(docker inspect -f '{{.State.Pid}}' $n); echo $BOARD_PASS | sudo -S -p '' kill -9 \$p" >/dev/null ;;
    2) local stream; stream=$(bssh "docker inspect -f '{{join .Args \" \"}}' $n" | sed -n 's|.*--url [^ ]*8554/\([^ ]*\).*|\1|p')
       local pids; pids=$(pgrep -f "ffmpeg .*rtsp://localhost:8554/${stream}\$")
       if [[ -n $stream && -n $pids ]]; then event "CHAOS camera outage 45 s on $stream (for $n): publisher SIGSTOP"; kill -STOP $pids; sleep 45; kill -CONT $pids; event "CHAOS camera $stream resumed"
       else event "CHAOS camera outage skipped (stream='$stream' pids='$pids')"; fi ;;
    3|5) local v=${VARIANTS[$((RANDOM % ${#VARIANTS[@]}))]}
       event "CHAOS dynamic parameter change $n -> variant $v (stop/rm/run with new args)"
       local r; r=$(bssh "cd $APP_DIR && ./recreate.sh $idx $v"); event "CHAOS result: ${r:-FAILED}"; [[ -z $r ]] && alert "re-create of $n ($v) failed" ;;
    4) event "CHAOS freeze 20 s (docker pause/unpause) $n"; bssh "docker pause $n; sleep 20; docker unpause $n" >/dev/null ;;
  esac
  sleep 30; sample   # extra sample right after the action so the recovery is visible
}

case "${1:-}" in
  run)
    rm -f "$STOPF"; echo "$$" > "$OUT/pid"; export KERR_BASE=${KERR_BASE:-2}; LAST_RT=0; LOW_STREAK=0; SAMPLE_N=0; CH_I=0
    LAST_RT=$(bssh 'rt=0; for c in $(docker ps -aq); do rt=$((rt + $(docker inspect -f "{{.RestartCount}}" $c))); done; echo $rt'); LAST_RT=${LAST_RT:-0}
    event "endurance start pid $$ chaos_every=${CHAOS_EVERY_MIN}min chaos=${CHAOS} baseline_restarts=$LAST_RT"
    while [[ ! -f $STOPF ]]; do
      SAMPLE_N=$((SAMPLE_N+1)); export SAMPLE_N
      sample
      if [[ $CHAOS == 1 && $SAMPLE_N -gt 10 && $(( SAMPLE_N % CHAOS_EVERY_MIN )) -eq 0 ]]; then chaos $CH_I; CH_I=$((CH_I+1)); fi
      sleep 60
    done
    event "endurance stopped"
    ;;
  stop) touch "$STOPF"; echo "stop requested" ;;
  once) SAMPLE_N=${SAMPLE_N:-1}; export SAMPLE_N; LAST_RT=0; LOW_STREAK=0; sample; tail -n 2 "$SAMPLES"; tail -n 1 "$LOGS/rates.tsv"; ls -la "$LOGS" ;;
  report)
    python3 - "$SAMPLES" "$ALERTS" "$EVENTS" <<'EOF'
import sys,csv,statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter="\t") if r.get("up") and r["up"]!="BOARD_UNREACHABLE"]
if not rows: print("no samples"); sys.exit()
f=lambda k:[float(r[k]) for r in rows if r.get(k) not in (None,"","?")]
print(f"samples={len(rows)} from {rows[0]['time']} to {rows[-1]['time']}")
print(f"up: min {min(f('up')):.0f} max {max(f('up')):.0f} | exited max {max(f('exited')):.0f} | restarts_total first {f('restarts_total')[0]:.0f} last {f('restarts_total')[-1]:.0f}")
print(f"rate_sum: mean {st.mean(f('rate_sum')):.1f} min {min(f('rate_sum')):.1f} | channels min {min(f('channels')):.0f} max {max(f('channels')):.0f}")
print(f"mem_used MB: first {f('mem_used_mb')[0]:.0f} last {f('mem_used_mb')[-1]:.0f} max {max(f('mem_used_mb')):.0f} | cma_free MB: min {min(f('cma_free_mb')):.0f} last {f('cma_free_mb')[-1]:.0f} | cache last {f('cache_mb')[-1]:.0f}")
print(f"temp C: max {max(f('temp_c')):.0f} | load1 max {max(f('load1')):.2f} | kerr last {f('kerr')[-1]:.0f} | docker_ok min {min(f('docker_ok')):.0f}")
low=[r for r in rows if r["low"] not in ("-","?")]; print(f"samples with low channels: {len(low)}")
try: print("--- alerts:", sum(1 for _ in open(sys.argv[2]))); print(open(sys.argv[2]).read()[-2500:])
except FileNotFoundError: print("--- alerts: 0")
try:
    ev=open(sys.argv[3]).read().splitlines(); ch=[e for e in ev if "CHAOS " in e and "result" not in e and "resumed" not in e]
    wd=[e for e in ev if "WATCHDOG" in e]
    print(f"--- events: {len(ev)} | chaos actions: {len(ch)} | watchdog restarts: {len(wd)}")
    import collections; print("    by type:", dict(collections.Counter(e.split("CHAOS ")[1].split(" ")[0]+" "+e.split("CHAOS ")[1].split(" ")[1] for e in ch)))
    print("\n".join(ev[-15:]))
except FileNotFoundError: pass
EOF
    ;;
  *) echo "usage: endurance.sh run|report|stop"; exit 2;;
esac
