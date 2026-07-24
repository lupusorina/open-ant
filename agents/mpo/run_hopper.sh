#!/bin/bash
# Launch 10 Hopper-v5 seeds in parallel.
set -euo pipefail
cd "$(dirname "$0")"

NUM_GPUS="${NUM_GPUS:-$(ls /proc/driver/nvidia/gpus 2>/dev/null | wc -l)}"
NUM_GPUS="${NUM_GPUS:-1}"

for seed in $(seq 0 9); do
  gpu=$((seed % NUM_GPUS))
  echo "Starting hopper seed=${seed} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 mpo_acme.py \
      --env_id Hopper-v5 \
      --exp_name mpo_hopper \
      --total_timesteps 2_000_000 \
      --seed "${seed}" \
      --cuda &
done
wait
echo "All hopper seeds finished."
