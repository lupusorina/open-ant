#!/bin/bash
set -euo pipefail
#
# Run dirs are written under RUNS_DIR as
# ${EXP_NAME}_<timestamp>_seed_<N>/weights_and_args.

RUNS_DIR="/data2/serenaliu_data/2sac"
SCRIPT="sac_cleanrl.py"

TOTAL_TIMESTEPS=190000
EXP_NAME="scratch_sim2"

# GPU to use for each seed, in order. Edit this list directly, e.g.
# GPUS=(0 2) to alternate GPU 0 and 2 across seeds.
GPUS=(1)

# Which seeds to run. Edit directly, e.g. SEEDS=(1 2 3).
SEEDS=(1 2 3 4 5 6)

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
        --render_mode rgb_array \
        --total_timesteps "${TOTAL_TIMESTEPS}" \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --runs_directory "${RUNS_DIR}" \
        --exp_name "${EXP_NAME}" \
        --num_envs 1 \
        --seed "${seed}" \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
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
