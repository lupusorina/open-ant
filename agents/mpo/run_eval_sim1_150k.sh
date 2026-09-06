#!/bin/bash
set -euo pipefail

# Usage:
#   bash run_eval_sim1_150k.sh
#
# All settings (which GPUs to use, how many jobs per GPU, etc.) are the plain
# variables below — edit them directly, no CLI flags.
#
# For every sim1 run under SIM_DIR (dirs named mpo_*_seed_<N>, produced by
# `runall_tmp.sh sim`, each trained to a 40k-step checkpoint), this loads that
# checkpoint and runs EVAL ONLY (no learner updates, no replay writes) for the
# same --total_timesteps as a normal continual-learning (Sim2) run, i.e. 150000,
# matching `run_continual` in runall_tmp.sh but with `--eval` added.
#
# Note on step counting: mpo_acme.py resets agent.global_step to 0 whenever
# --eval is combined with --weights_path (see main() in mpo_acme.py), so each
# eval run here actually executes 150000 env steps from the loaded checkpoint,
# rather than resuming the checkpoint's own step counter up to 150000.
#
# Eval run dirs are written directly under SIM_DIR (alongside the existing
# mpo_*_seed_* / continuous_mpo_*_seed_* dirs), as
# eval_sim1_150k_<timestamp>_seed_<N>/weights_and_args, so they live in the
# same place as the rest of that seed's runs. Logs go to a separate
# SIM_DIR/eval_sim1_150k_logs/ folder so they don't clutter the run dirs.

SIM_DIR="/data2/serenaliu_data/2dmpo"
SCRIPT="mpo_acme.py"
ENSEMBLE=1

EVAL_TOTAL_TIMESTEPS=110000
EVAL_EXP_NAME="eval_sim1_150k"
EVAL_RUNS_DIR="${SIM_DIR}"
LOG_DIR="${SIM_DIR}/eval_sim1_150k_logs"

# GPUs to use. Edit this list directly, e.g. GPUS=(0 2) to skip 1 and 3.
GPUS=(0 1)

# How many eval jobs to run concurrently on each GPU.
JOBS_PER_GPU=5

# Which seeds to run. Leave empty () to run every mpo_*_seed_* dir found in
# SIM_DIR. Otherwise list only the ones you want, e.g. SEEDS=(1 2 3 8).
SEEDS=(1 2 3 4 8 9 12 13 14 15 16 17)

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
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --runs_directory "${EVAL_RUNS_DIR}" \
        --exp_name "${EVAL_EXP_NAME}" \
        --seed "${seed}" \
        --weights_path "${weights_path}" \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
        --critic_type categorical \
        --eval \
        --cuda \
        > "${log_file}" 2>&1
    then
        touch "${FAIL_DIR}/${seed}"
    fi
}

mapfile -t SIM_DIRS < <(find "${SIM_DIR}" -maxdepth 1 -type d -name "mpo_*_seed_*" | sort)

if [ "${#SIM_DIRS[@]}" -eq 0 ]; then
    echo "ERROR: No mpo_*_seed_* directories found under ${SIM_DIR}"
    exit 1
fi

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
