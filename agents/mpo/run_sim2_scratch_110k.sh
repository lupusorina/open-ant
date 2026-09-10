#!/bin/bash
set -euo pipefail
#
# Run dirs are written under RUNS_DIR as
# ${EXP_NAME}_<timestamp>_seed_<N>/weights_and_args.

RUNS_DIR="/data2/serenaliu_data/2empo"
SCRIPT="mpo_acme.py"
ENSEMBLE=3

TOTAL_TIMESTEPS=190000
EXP_NAME="scratch_sim2"

# GPU to use for each seed, in order. Edit this list directly, e.g.
# GPUS=(0 2) to alternate GPU 0 and 2 across seeds.
GPUS=(1)

# Which seeds to run. Edit directly, e.g. SEEDS=(1 2 3).
SEEDS=(20 21 22)
#for empo
# 10 11 12 15
# 23 24 25 26 27 28 29 30
# 16 17 18 19

# for dmpo
# 1 2 3 4 5 6 7 8 9 10
# 11 12 13 14 15
# 16 17 18 19 20
# 21 22 23 24 25 26 27 28 29 30

mkdir -p "${RUNS_DIR}"

run_scratch () {
    local seed="$1"
    local gpu="$2"

    echo "=========================================="
    echo "Running from-scratch Sim2 for seed ${seed} on GPU ${gpu}"
    echo "Output parent dir: ${RUNS_DIR}"
    echo "Exp name: ${EXP_NAME}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="${gpu}" python3 "${SCRIPT}" \
        --ensemble "${ENSEMBLE}" \
        --render_mode rgb_array \
        --total_timesteps "${TOTAL_TIMESTEPS}" \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --runs_directory "${RUNS_DIR}" \
        --exp_name "${EXP_NAME}" \
        --seed "${seed}" \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
        --critic_type scalar \
        --cuda

    echo "From-scratch Sim2 complete for seed ${seed}."
}

gpu_idx=0

for seed in "${SEEDS[@]}"; do
    gpu="${GPUS[$(( gpu_idx % ${#GPUS[@]} ))]}"
    gpu_idx=$(( gpu_idx + 1 ))

    run_scratch "${seed}" "${gpu}"
done

echo "=========================================="
echo "All from-scratch sim2 runs completed."
