#!/bin/bash
set -euo pipefail

# Usage:
#   bash run_eval_sim1_150k.sh ant
#   bash run_eval_sim1_150k.sh walker
#
# All other settings (which GPUs to use, how many jobs per GPU, seeds, etc.)
# are the plain variables below — edit them directly, no CLI flags besides
# the ant/walker mode.
#
# For every sim1 run found under SIM_DIR (dirs named ${SIM1_EXP_NAME}_*_seed_<N>,
# produced by that mode's Sim1 training), this loads that checkpoint and runs
# EVAL ONLY (no learner updates, no replay writes) for EVAL_TOTAL_TIMESTEPS
# steps using the mode's Sim2 model, i.e.:
#   - ant:    matches `run_continual` in runall_tmp.sh (150000 steps,
#             ant_with_camera_after_sys_id_real_less_aggresive.xml) but with
#             --eval added.
#   - walker: matches the Sim2 stage in run_walkerall.sh (2_500_000 steps,
#             walker2d_sim2_massfric.xml) but with --eval added, so the
#             sim1-trained-to-1M-steps checkpoint is evaluated on the sim2
#             model with no learning.
#
# Note on step counting: mpo_acme.py (and friends) reset agent.global_step to
# 0 whenever --eval is combined with --weights_path (see main() in
# mpo_acme.py), so each eval run here actually executes EVAL_TOTAL_TIMESTEPS
# env steps from the loaded checkpoint, rather than resuming the checkpoint's
# own step counter.
#
# Eval run dirs are written directly under SIM_DIR (alongside the existing
# sim1/sim2 run dirs), as eval_sim1_150k_<timestamp>_seed_<N>/weights_and_args,
# so they live in the same place as the rest of that seed's runs. Logs go to a
# separate SIM_DIR/eval_sim1_150k_logs/ folder so they don't clutter the run
# dirs.

MODE="${1:-}"
case "${MODE}" in
    ant|walker) ;;
    *)
        echo "Usage: bash run_eval_sim1_150k.sh {ant|walker}" >&2
        exit 1
        ;;
esac

EVAL_EXP_NAME="eval_sim1_150k"

# GPUs to use. Edit this list directly, e.g. GPUS=(0 2) to skip 1 and 3.
GPUS=(1 2 3)

# How many eval jobs to run concurrently on each GPU.
JOBS_PER_GPU=5

# Which seeds to run. Leave empty () to run every ${SIM1_EXP_NAME}_*_seed_*
# dir found in SIM_DIR. Otherwise list only the ones you want, e.g. SEEDS=(1 2 3 8).
SEEDS=(13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

if [ "${MODE}" == "ant" ]; then
    SIM_DIR="/data2/serenaliu_data/1ant_sim_all/2qrdqn_avg4actions"
    SCRIPT="mpo_qrdqn.py"
    ENSEMBLE=1
    ENV_ID="SimEmbodiedAnt"
    MODEL_PATH="../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml"
    CRITIC_TYPE="quantile"
    EVAL_TOTAL_TIMESTEPS=110000
    SIM1_EXP_NAME="mpo"
    DT_ARGS=(--dt 0.12)
else
    SIM_DIR="/data2/serenaliu_data/2empo_walker_impfast_spi192"
    SCRIPT="mpo_acme.py"
    ENSEMBLE=3
    ENV_ID="Walker2d-v5"
    MODEL_PATH="../../sim/assets/walker2d_sim2_massfric.xml"
    CRITIC_TYPE="scalar"
    EVAL_TOTAL_TIMESTEPS=1500000
    SIM1_EXP_NAME="mpo_walker"
    DT_ARGS=()
fi

EVAL_RUNS_DIR="${SIM_DIR}"
LOG_DIR="${SIM_DIR}/eval_sim1_150k_logs"

MAX_PARALLEL=$(( ${#GPUS[@]} * JOBS_PER_GPU ))

mkdir -p "${EVAL_RUNS_DIR}" "${LOG_DIR}"

FAIL_DIR=$(mktemp -d)
trap 'rm -rf "${FAIL_DIR}"' EXIT

run_eval () {
    local seed="$1"
    local weights_path="$2"
    local gpu="$3"
    local log_file="${LOG_DIR}/seed_${seed}.log"

    echo "Launching eval for seed ${seed} on GPU ${gpu} (log: ${log_file})"

    if ! CUDA_VISIBLE_DEVICES="${gpu}" python3 "${SCRIPT}" \
        --ensemble "${ENSEMBLE}" \
        --render_mode rgb_array \
        --total_timesteps "${EVAL_TOTAL_TIMESTEPS}" \
        "${DT_ARGS[@]}" \
        --env_id "${ENV_ID}" \
        --runs_directory "${EVAL_RUNS_DIR}" \
        --exp_name "${EVAL_EXP_NAME}" \
        --seed "${seed}" \
        --weights_path "${weights_path}" \
        --model_path "${MODEL_PATH}" \
        --critic_type "${CRITIC_TYPE}" \
        --eval \
        --cuda \
        > "${log_file}" 2>&1
    then
        touch "${FAIL_DIR}/${seed}"
    fi
}

mapfile -t SIM_DIRS < <(find "${SIM_DIR}" -maxdepth 1 -type d -name "${SIM1_EXP_NAME}_*_seed_*" | sort)

if [ "${#SIM_DIRS[@]}" -eq 0 ]; then
    echo "ERROR: No ${SIM1_EXP_NAME}_*_seed_* directories found under ${SIM_DIR}"
    exit 1
fi

echo "Mode: ${MODE}"
echo "Found ${#SIM_DIRS[@]} sim1 checkpoints under ${SIM_DIR}."
if [ "${#SEEDS[@]}" -gt 0 ]; then
    echo "Restricting to seeds: ${SEEDS[*]}"
fi
echo "GPUs: ${GPUS[*]} (${JOBS_PER_GPU} jobs/gpu, ${MAX_PARALLEL} total in parallel)"
echo "Running eval-only to ${EVAL_TOTAL_TIMESTEPS} steps."
echo "Eval outputs -> ${EVAL_RUNS_DIR}/${EVAL_EXP_NAME}_*_seed_<N>/"
echo "=========================================="

gpu_idx=0

for sim_run_dir in "${SIM_DIRS[@]}"; do
    base=$(basename "${sim_run_dir}")
    seed="${base##*_seed_}"

    if [ "${#SEEDS[@]}" -gt 0 ]; then
        keep=0
        for wanted in "${SEEDS[@]}"; do
            if [ "${wanted}" == "${seed}" ]; then
                keep=1
                break
            fi
        done
        if [ "${keep}" -eq 0 ]; then
            continue
        fi
    fi

    weights_path="${sim_run_dir}/weights_and_args"
    if [ ! -d "${weights_path}" ]; then
        echo "WARNING: skipping seed ${seed}, no weights_and_args in ${sim_run_dir}"
        continue
    fi

    gpu="${GPUS[$(( gpu_idx % ${#GPUS[@]} ))]}"
    gpu_idx=$(( gpu_idx + 1 ))

    run_eval "${seed}" "${weights_path}" "${gpu}" &

    # Throttle: block until a slot frees up once we hit MAX_PARALLEL.
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
        wait -n
    done
done

# Wait for all remaining background jobs to finish.
wait

failures=$(find "${FAIL_DIR}" -mindepth 1 | wc -l)

echo "=========================================="
if [ "${failures}" -eq 0 ]; then
    echo "All eval runs completed successfully."
else
    echo "${failures} eval run(s) failed (seeds: $(ls "${FAIL_DIR}")). Check logs in ${LOG_DIR}"
    exit 1
fi
