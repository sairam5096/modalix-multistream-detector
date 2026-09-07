#!/bin/bash
# Robust launcher for the on-device detector: fully detached so an SSH disconnect
# does not kill it. Usage: launch.sh <config.yaml>
cd "$(dirname "$0")/.." || exit 1
CFG="${1:-configs/example-48x5fps.yaml}"
nohup setsid ./shared-multi-model-detector --config "$CFG" >./detector.log 2>&1 </dev/null &
sleep 1
echo "launched pid $! with $CFG  (log: som/detector.log)"
