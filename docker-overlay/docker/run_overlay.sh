#!/bin/bash
# Start ONE C++ on-SOM-overlay detector container (binary expected at docker/build/overlay-detector).
# Usage: ./run_overlay.sh N RTSP_URL CH [app args...] [-- docker args...]
#   e.g. ./run_overlay.sh 3 rtsp://host:8554/cam3 2 --fps 5 --decode yolov6 --model models2/yolov6n_mpk.tar.gz
# Scratch image + read-only board mounts; MODELS_DIR (default /data/models) is mounted at models2.
set -eo pipefail
cd "$(dirname "$0")"
app_dir=$(pwd -P)
[[ -f demo.env ]] && . ./demo.env   # INSIGHT_HOST, MEDIA_HOST, MODELS_DIR, IMAGE
: "${INSIGHT_HOST:?set INSIGHT_HOST (Neat Insight address) in demo.env or the environment}"
N=${1:?container number}; URL=${2:?rtsp url}; CH=${3:?insight channel}; shift 3
APP_ARGS=(); DOCKER_ARGS=(); mode=app
for x in "$@"; do if [[ $x == "--" ]]; then mode=docker; continue; fi; if [[ $mode == app ]]; then APP_ARGS+=("$x"); else DOCKER_ARGS+=("$x"); fi; done
set --
[[ -x build/overlay-detector ]] || { echo "missing build/overlay-detector" >&2; exit 1; }
for path in /usr/lib /lib /etc/ld.so.cache /etc/alternatives \
  /usr/share/sima-neat /usr/share/glib-2.0 /bin/sh /usr/bin/tar /usr/bin/gzip; do
  set -- "$@" --mount "type=bind,src=$path,dst=$path,readonly"
done
exec docker run -d --restart unless-stopped --name "neat-ovc-$N" --network host --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  --log-driver json-file --log-opt max-size=10m --log-opt max-file=3 \
  --security-opt seccomp=unconfined \
  --device /dev/cvu --device /dev/mla \
  --device /dev/dma_heap/linux,cma --device /dev/dma_heap/simaai,dms \
  --device /dev/allegroDecodeIP --device /dev/allegroIP \
  --device /dev/simaai-mem \
  "$@" "${DOCKER_ARGS[@]}" \
  --mount "type=bind,src=$app_dir/build/overlay-detector,dst=/opt/neat-app/overlay-detector,readonly" \
  --mount "type=bind,src=$app_dir/models,dst=/opt/neat-app/models,readonly" \
  --mount "type=bind,src=${MODELS_DIR:-/data/models},dst=/opt/neat-app/models2,readonly" \
  --mount "type=bind,src=$app_dir/labels.txt,dst=/opt/neat-app/labels.txt,readonly" \
  --entrypoint /opt/neat-app/overlay-detector \
  ${IMAGE:-neat-overlay:mounted} --url "$URL" --channel "$CH" --host "$INSIGHT_HOST" "${APP_ARGS[@]}"
