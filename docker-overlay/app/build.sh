#!/bin/bash
# Cross-build inside the SiMa Neat SDK container (aarch64 sysroot at /opt/toolchain/aarch64/modalix).
set -euo pipefail
cd "$(dirname "$0")"
SYSROOT=${SYSROOT:-/opt/toolchain/aarch64/modalix}
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DCMAKE_SYSROOT="$SYSROOT" -DCMAKE_FIND_ROOT_PATH="$SYSROOT" \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY
cmake --build build -j"$(nproc)"
echo "built: $(pwd)/build/overlay-detector  -> copy to docker/build/overlay-detector on the board"
