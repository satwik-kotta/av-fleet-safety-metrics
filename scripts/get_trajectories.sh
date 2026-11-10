#!/usr/bin/env bash
# Downloads the NGSIM US-101 vehicle trajectories (FHWA, public domain), in the
# Savitzky-Golay smoothed version (window 11) published at
# https://github.com/Rim-El-Ballouli/NGSIM-US-101-trajectory-dataset-smoothing
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/ngsim/raw
if [ ! -d data/ngsim/src ]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/Rim-El-Ballouli/NGSIM-US-101-trajectory-dataset-smoothing.git data/ngsim/src
  git -C data/ngsim/src sparse-checkout set --no-cone 'smoothed-dataset/window-11/*.zip'
  git -C data/ngsim/src checkout -q HEAD
fi
for z in data/ngsim/src/smoothed-dataset/window-11/*.zip; do unzip -o -q "$z" -d data/ngsim/raw; done
ls -la data/ngsim/raw
