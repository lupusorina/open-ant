#!/bin/bash
set -euo pipefail
#
# Train Sim2 from scratch (no Sim1 warm-start weights) for a chosen env.
#
# Usage:
#   bash run_sim2_scratch_110k.sh {ant|humanoid|walker} [total_timesteps]
#
# Run dirs are written under RUNS_DIR as
# ${EXP_NAME}_<timestamp>_seed_<N>/weights_and_args.

ENV_NAME="${1:-ant}"
SCRIPT="mpo_acme.py"
ENSEMBLE=3
CRITIC_TYPE="scalar"

# GPU to use for each seed, in order. Edit this list directly, e.g.
# GPUS=(0 2) to alternate GPU 0 and 2 across seeds.
GPUS=(1)

# Which seeds to run. Edit directly, e.g. SEEDS=(1 2 3).
SEEDS=(1)
#for empo
# 10 11 12 15
# 23 24 25 26 27 28 29 30
# 16 17 18 19

# for dmpo
# 1 2 3 4 5 6 7 8 9 10
# 11 12 13 14 15
# 16 17 18 19 20
# 21 22 23 24 25 26 27 28 29 30

# -----------------------------------------------------------------------------
# Per-env config: env_id, Sim2 model xml, runs dir, exp name, default
# total_timesteps, and any extra mpo_acme.py flags this env needs.
# -----------------------------------------------------------------------------
EXTRA_ARGS=()

case "${ENV_NAME}" in
    ant)
        ENV_ID="SimEmbodiedAnt"
        MODEL_PATH="../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml"
        RUNS_DIR="/data2/serenaliu_data/2empo"
        EXP_NAME="scratch_sim2"
        DEFAULT_TOTAL_TIMESTEPS=190000
        EXTRA_ARGS=(--dt 0.12)
        ;;
    humanoid)
        ENV_ID="Humanoid-v5"
        MODEL_PATH="../../sim/assets/humanoid_sim2.xml"
        RUNS_DIR="/data2/serenaliu_data/2empo_humanoid_SPI128"
        EXP_NAME="scratch_sim2_humanoid"
        DEFAULT_TOTAL_TIMESTEPS=5_000_000
        EXTRA_ARGS=(--gamma 0.99 --dual_lr 0.005 --policy_init_scale 0.5 --samples_per_insert 128)
        ;;
    walker)
        ENV_ID="Walker2d-v5"
        MODEL_PATH="../../sim/assets/walker2d_sim2_massfric.xml"
        RUNS_DIR="/data2/serenaliu_data/2empo_walker_impfast_spi192"
        EXP_NAME="scratch_sim2_walker"
        DEFAULT_TOTAL_TIMESTEPS=2_500_000
        EXTRA_ARGS=(--gamma 0.99 --dual_lr 0.005 --policy_init_scale 0.5 --samples_per_insert 192)
        ;;
    *)
        echo "Usage: bash run_sim2_scratch_110k.sh {ant|humanoid|walker} [total_timesteps]" >&2
        exit 1
        ;;
esac

TOTAL_TIMESTEPS="${2:-${DEFAULT_TOTAL_TIMESTEPS}}"

mkdir -p "${RUNS_DIR}"

run_scratch () {
    local seed="$1"
    local gpu="$2"

    echo "=========================================="
    echo "Running from-scratch Sim2 (${ENV_NAME}) for seed ${seed} on GPU ${gpu}"
    echo "Output parent dir: ${RUNS_DIR}"
    echo "Exp name: ${EXP_NAME}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="${gpu}" python3 "${SCRIPT}" \
        --ensemble "${ENSEMBLE}" \
        --render_mode rgb_array \
        --total_timesteps "${TOTAL_TIMESTEPS}" \
        --env_id "${ENV_ID}" \
        --runs_directory "${RUNS_DIR}" \
        --exp_name "${EXP_NAME}" \
        --seed "${seed}" \
        --model_path "${MODEL_PATH}" \
        --critic_type "${CRITIC_TYPE}" \
        --cuda \
        "${EXTRA_ARGS[@]}"

    echo "From-scratch Sim2 complete for seed ${seed}."
}

gpu_idx=0

for seed in "${SEEDS[@]}"; do
    gpu="${GPUS[$(( gpu_idx % ${#GPUS[@]} ))]}"
    gpu_idx=$(( gpu_idx + 1 ))

    run_scratch "${seed}" "${gpu}"
done

echo "=========================================="
echo "All from-scratch Sim2 (${ENV_NAME}) runs completed."
