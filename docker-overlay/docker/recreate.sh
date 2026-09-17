#!/bin/bash
# Re-create overlay container neat-ovc-N with a runtime-parameter variant.
# Dumps the old container's logs to /data/endurance_logs/ before removing it.
#   recreate.sh N default|yolov6n|fps10|res360|v6fps10|v6res360|v6sw
# Stream names below (mystream_5fpsNN, mystream_10fpsNN, mystream_360p10_42) match our test bench; edit to match yours.
set -u
N=${1:?N}; V=${2:-default}; name=neat-ovc-$N; idx=$(printf %02d "$N"); ch=$((N-1))
cd "$(dirname "$0")" || exit 1
[[ -f demo.env ]] && . ./demo.env
HOST=${MEDIA_HOST:?set MEDIA_HOST (RTSP server address) in demo.env or the environment}
case $V in
  default) set -- "rtsp://$HOST:8554/mystream_5fps$idx" $ch --fps 5 ;;
  yolov6n) set -- "rtsp://$HOST:8554/mystream_5fps$idx" $ch --fps 5 --decode yolov6 --model models2/yolov6n_mpk.tar.gz ;;
  fps10)   set -- "rtsp://$HOST:8554/mystream_10fps$idx" $ch --fps 10 ;;
  res360)  set -- "rtsp://$HOST:8554/mystream_360p10_42" $ch --fps 10 --width 640 --height 360 ;;
  v6fps10)  set -- "rtsp://$HOST:8554/mystream_10fps$idx" $ch --fps 10 --decode yolov6 --model models2/yolov6n_mpk.tar.gz ;;
  v6res360) set -- "rtsp://$HOST:8554/mystream_360p10_42" $ch --fps 10 --width 640 --height 360 --decode yolov6 --model models2/yolov6n_mpk.tar.gz ;;
  v6sw)     set -- "rtsp://$HOST:8554/mystream_5fps$idx" $ch --fps 5 --decode yolov6 --model models2/yolov6n_mpk.tar.gz --encoder sw ;;
  *) echo "unknown variant $V" >&2; exit 2 ;;
esac
mkdir -p /data/endurance_logs
if docker inspect "$name" >/dev/null 2>&1; then
  { echo "===== $(date -u +%FT%TZ) $name before re-create to variant=$V args=$(docker inspect -f '{{join .Args " "}}' "$name")"; docker logs -t "$name" 2>&1 | tail -n 400; } >> "/data/endurance_logs/$name.log"
  docker stop -t 5 "$name" >/dev/null 2>&1; sleep 6; docker rm "$name" >/dev/null 2>&1
fi
./run_overlay.sh "$N" "$@" >/dev/null && echo "recreated $name variant=$V args=$*"
