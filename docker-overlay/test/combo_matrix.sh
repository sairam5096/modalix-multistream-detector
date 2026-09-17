#!/bin/bash
# Run combo_sweep.sh for several K values, with a graceful teardown + reboot between them.
#   setsid nohup ./combo_matrix.sh "1 2 4 8 16" &
set -uo pipefail
cd "$(dirname "$0")"
KS=${1:-"1 2 4 8 16"}
BOARD=${BOARD:?set BOARD to the SOM IP}
BOARD_PASS=${BOARD_PASS:?set BOARD_PASS to the board ssh password}
S=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=8)
bssh() { timeout 300 sshpass -p "$BOARD_PASS" ssh "${S[@]}" sima@"$BOARD" "$@" 2>/dev/null; }
mkdir -p results/combo; LOG=results/combo/matrix.log
clean_reboot() {
  bssh 'for c in $(docker ps --format "{{.Names}}" | sort -t- -k3 -n); do docker stop -t 5 $c >/dev/null; sleep 4; done; docker rm $(docker ps -aq) >/dev/null 2>&1; sync; echo $BOARD_PASS | sudo -S -p "" reboot'
  sleep 100
  for i in $(seq 1 40); do bssh 'docker info >/dev/null 2>&1 && [ "$(docker ps -aq | wc -l)" = 0 ] && echo ready' | grep -q ready && break; sleep 15; done
  sleep 45
}
for K in $KS; do
  echo "$(date -u +%FT%TZ) === K=$K: clean reboot" >> $LOG
  clean_reboot
  echo "$(date -u +%FT%TZ) === K=$K: sweep" >> $LOG
  ./combo_sweep.sh "$K" 48 >> $LOG 2>&1
done
echo "$(date -u +%FT%TZ) MATRIX DONE" >> $LOG
