#!/bin/bash
# Launch 10 continual-learning Hopper-v5 seeds simultaneously on GPU 2.

set -euo pipefail
cd "$(dirname "$0")"

SCRIPT="mpo_acme.py"
GPU_ID="${GPU_ID:-2}"
OUTPUT_DIR="/data2/serenaliu_data/continual_mpo_walker2d_massfric"
MODEL_PATH="/home/serenaliu/caltech_linc_home/open-ant/sim/assets/walker2d_sim2_massfric.xml"
CONT_EXP_NAME="continual_mpo_walker"
RUNS_ROOT="/data2/sorina_data/runs_July_24"
for seed in $(seq 0 3); do
    seed_dir="$(
        find "${RUNS_ROOT}" \
            -maxdepth 1 \
            -type d \
            -name "mpo_walker_*_seed_${seed}" |
        sort |
        tail -n 1
    )"

    weights_path="${seed_dir}/weights_and_args"
    echo "Starting seed ${seed} on GPU ${GPU_ID}"
    echo "Run directory: ${seed_dir}"
    echo "Checkpoint: ${weights_path}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT}" \
        --env_id Walker2d-v5 \
        --exp_name "${CONT_EXP_NAME}" \
        --total_timesteps 5_000_000 \
        --seed "${seed}" \
        --runs_directory "${OUTPUT_DIR}" \
        --weights_path "${weights_path}" \
        --model_path "${MODEL_PATH}" \
        --cuda &
done
wait
echo "All continual-learning seeds finished."