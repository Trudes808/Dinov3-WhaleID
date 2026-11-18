#!/bin/bash
# one GPU turn: flock, then at most 20 minutes for the commands given (each line of $1 is a command run in this folder)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8
export TORCH_HOME=/home/sat3737/NOAA_whales/moonshot_cache/local_match/models/torch HF_HOME=/home/sat3737/NOAA_whales/moonshot_cache/local_match/models/hf
exec flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 1200 bash -c "$1"
