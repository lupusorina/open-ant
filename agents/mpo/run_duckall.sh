#!/usr/bin/env bash
# Run independent walker Sim1 -> Sim2 pipelines.
# At most one pipeline runs on each selected GPU.
#
# Usage:
#   bash run_duckall.sh sim                     # Sim1 only
#   bash run_duckall.sh sim_resume               # Resume the latest Sim1 run (weights + optimizers +
#                                                #   replay buffer + dual vars) for SIM1_RESUME_STEPS more steps
#   bash run_duckall.sh sim_continual_learning   # Sim2 only, auto-detects existing Sim1 run per seed
#   bash run_duckall.sh sim_then_continual       # Sim1 -> Sim2 (default)

set -euo pipefail
cd "$(dirname "$0")"

SCRIPT="mpo_acme.py"

GPU_LIST=(0)

SEEDS=(1)
RUN_MODE="${1:-sim_then_continual}"

case "${RUN_MODE}" in
    sim|sim_resume|sim_continual_learning|sim_then_continual) ;;
    *)
        echo "Usage: bash run_duckall.sh {sim|sim_resume|sim_continual_learning|sim_then_continual}" >&2
        exit 1
        ;;
esac
SIM1_EXP_NAME="mpo_duck"
SIM1_TOTAL_TIMESTEPS="1_000_000"
# How many MORE steps `sim_resume` trains beyond whatever step the latest
# Sim1 checkpoint left off at. Plain digits (no "_" separators) — this value
# is used in bash arithmetic below, which (unlike Python's int()) can't parse
# underscore-grouped literals.
SIM1_RESUME_STEPS="${SIM1_RESUME_STEPS:-5000000}"
GLOBAL_TOTAL_TIMESTEPS="2_500_000"

RUNS_DIR="/data2/serenaliu_data/2empo_duck"

MODEL1_PATH="../../sim/assets/robot_walk.xml"
MODEL2_PATH="../../sim/assets/walker2d_sim2_massfric.xml"

CRITIC_TYPE="scalar"
ENSEMBLE=3

CONT_EXP_NAME="continual_mpo_duck"



run_seed_pipeline() {
    local seed="$1"
    local physical_gpu="$2"
    local sim1_run_dir=""
    local weights_path=""

    # -------------------------------------------------------------------------
    # Resume Sim1 in place: same weights, optimizers, dual variables, replay
    # buffer, and global_step as the latest Sim1 run for this seed (mpo_acme.py's
    # --resume_in_place / load_checkpoint loads all of it) — just SIM1_RESUME_STEPS
    # more steps on top of wherever the latest checkpoint left off.
    # -------------------------------------------------------------------------
    if [[ "${RUN_MODE}" == "sim_resume" ]]; then
        sim1_run_dir="$(
            find "${RUNS_DIR}" \
                -maxdepth 1 \
                -type d \
                -name "${SIM1_EXP_NAME}_*_seed_${seed}" \
                -printf '%T@ %p\n' 2>/dev/null |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-
        )"

        if [[ -z "${sim1_run_dir}" ]]; then
            echo "ERROR: no existing Sim1 run found for seed ${seed} matching ${SIM1_EXP_NAME}_*_seed_${seed} in ${RUNS_DIR}" >&2
            return 1
        fi

        weights_path="${sim1_run_dir}/weights_and_args"
        if [[ ! -d "${weights_path}" ]]; then
            echo "ERROR: Missing weights directory: ${weights_path}" >&2
            return 1
        fi

        local last_ckpt_step
        last_ckpt_step="$(
            find "${weights_path}" -maxdepth 1 -name 'checkpoint_*.pth' -printf '%f\n' 2>/dev/null |
            sed -E 's/checkpoint_([0-9]+)\.pth/\1/' |
            sort -n |
            tail -n 1
        )"
        if [[ -z "${last_ckpt_step}" ]]; then
            echo "ERROR: no checkpoint_*.pth files found in ${weights_path}" >&2
            return 1
        fi

        local resume_total_timesteps=$((last_ckpt_step + SIM1_RESUME_STEPS))

        echo
        echo "Seed ${seed}: resuming Sim1 from ${sim1_run_dir}"
        echo "Seed ${seed}: latest checkpoint step ${last_ckpt_step} -> training to total_timesteps=${resume_total_timesteps} (+${SIM1_RESUME_STEPS})"

        CUDA_VISIBLE_DEVICES="${physical_gpu}" python3 "${SCRIPT}" \
            --env_id Microduck-v0 \
            --render_mode rgb_array \
            --exp_name "${SIM1_EXP_NAME}" \
            --total_timesteps "${resume_total_timesteps}" \
            --seed "${seed}" \
            --runs_directory "${RUNS_DIR}" \
            --cuda \
            --model_path "${MODEL1_PATH}" \
            --critic_type "${CRITIC_TYPE}" \
            --ensemble "${ENSEMBLE}" \
            --gamma 0.99 \
            --dual_lr 0.005 \
            --policy_init_scale 0.5 \
            --samples_per_insert 192 \
            --resume_in_place \
            --weights_path "${weights_path}"

        echo "Seed ${seed}: Sim1 resume finished."
        return
    fi

    # -------------------------------------------------------------------------
    # Sim1
    # -------------------------------------------------------------------------
    if [[ "${RUN_MODE}" != "sim_continual_learning" ]]; then
        echo
        echo "Seed ${seed}: starting Sim1 on physical GPU ${physical_gpu}"

        local marker
        marker="$(mktemp)"
        touch "${marker}"

        CUDA_VISIBLE_DEVICES="${physical_gpu}" python3 "${SCRIPT}" \
            --env_id Microduck-v0 \
            --render_mode rgb_array \
            --exp_name "${SIM1_EXP_NAME}" \
            --total_timesteps "${SIM1_TOTAL_TIMESTEPS}" \
            --seed "${seed}" \
            --runs_directory "${RUNS_DIR}" \
            --cuda \
            --model_path "${MODEL1_PATH}" \
            --critic_type "${CRITIC_TYPE}" \
            --ensemble "${ENSEMBLE}" \
            --gamma 0.99 \
            --dual_lr 0.005 \
            --policy_init_scale 0.5 \
            --samples_per_insert 192


        sim1_run_dir="$(
            find "${RUNS_DIR}" \
                -maxdepth 1 \
                -type d \
                -name "${SIM1_EXP_NAME}_*_seed_${seed}" \
                -newer "${marker}" \
                -printf '%T@ %p\n' 2>/dev/null |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-
        )"

        rm -f "${marker}"

        echo "Seed ${seed}: Sim1 finished."
        echo "Sim1 run: ${sim1_run_dir}"

        # Stop here for Sim1-only mode.
        if [[ "${RUN_MODE}" == "sim" ]]; then
            return
        fi
    fi

    # -------------------------------------------------------------------------
    # Find existing Sim1 run for Sim2-only (sim_continual_learning) mode
    # -------------------------------------------------------------------------
    if [[ "${RUN_MODE}" == "sim_continual_learning" ]]; then
        sim1_run_dir="$(
            find "${RUNS_DIR}" \
                -maxdepth 1 \
                -type d \
                -name "${SIM1_EXP_NAME}_*_seed_${seed}" \
                -printf '%T@ %p\n' 2>/dev/null |
            sort -nr |
            head -n 1 |
            cut -d' ' -f2-
        )"

        echo "Seed ${seed}: using existing Sim1 run:"
        echo "  ${sim1_run_dir}"
    fi

    weights_path="${sim1_run_dir}/weights_and_args"

    if [[ ! -d "${weights_path}" ]]; then
        echo "ERROR: Missing weights directory: ${weights_path}" >&2
        return 1
    fi

    # -------------------------------------------------------------------------
    # Sim2
    # -------------------------------------------------------------------------
    echo
    echo "Seed ${seed}: starting Sim2"
    echo "Loading: ${weights_path}"
    echo "Loading: ${MODEL2_PATH}"

    CUDA_VISIBLE_DEVICES="${physical_gpu}" python3 "${SCRIPT}" \
        --env_id Walker2d-v5 \
        --render_mode rgb_array \
        --exp_name "${CONT_EXP_NAME}" \
        --total_timesteps "${GLOBAL_TOTAL_TIMESTEPS}" \
        --seed "${seed}" \
        --runs_directory "${RUNS_DIR}" \
        --weights_path "${weights_path}" \
        --model_path "${MODEL2_PATH}" \
        --cuda \
        --critic_type "${CRITIC_TYPE}" \
        --ensemble "${ENSEMBLE}" \
        --gamma 0.99 \
        --dual_lr 0.005 \
        --policy_init_scale 0.5 \
        --samples_per_insert 192
    echo
    echo "Seed ${seed}: Sim1 and Sim2 both finished on GPU ${physical_gpu}."

} 

    


run_gpu_worker() {
    local worker_index="$1"
    local physical_gpu="${GPU_LIST[$worker_index]}"
    local num_gpus="${#GPU_LIST[@]}"

    echo "Starting worker ${worker_index} on physical GPU ${physical_gpu}"

    # This worker handles every num_gpus-th seed.
    for ((seed_index=worker_index; seed_index<${#SEEDS[@]}; seed_index+=num_gpus)); do
        local seed="${SEEDS[$seed_index]}"

        run_seed_pipeline "${seed}" "${physical_gpu}"
    done

    echo "GPU worker ${physical_gpu} finished all assigned seeds."
}


# =============================================================================
# Start one worker per selected GPU
# =============================================================================

worker_pids=()

for worker_index in "${!GPU_LIST[@]}"; do
    run_gpu_worker "${worker_index}" &
    worker_pids+=("$!")

    echo "Launched GPU ${GPU_LIST[$worker_index]} worker with PID $!"
done


# =============================================================================
# Wait for all GPU workers
# =============================================================================

failed=0

for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
        echo "ERROR: GPU worker PID ${pid} failed." >&2
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    echo "One or more seed pipelines failed." >&2
    exit 1
fi

echo
echo "============================================================"
echo "All walker Sim1 -> Sim2 pipelines finished."
echo "GPUs used: ${GPU_LIST[*]}"
echo "============================================================"
