#!/usr/bin/env bash
# Run independent walker Sim1 -> Sim2 pipelines.
# At most one pipeline runs on each selected GPU.

set -euo pipefail
cd "$(dirname "$0")"

SCRIPT="mpo_acme.py"

# Number of GPUs to use: GPUs 0..NUM_GPUS-1 each get their own worker (see
# run_gpu_worker below). This matches how Slurm exposes GPUs: an allocated
# GPU is always local index 0, 1, ... regardless of its physical index on the
# node, so NUM_GPUS should just match --gres/--gpus-per-task in the sbatch
# script. Override at the command line, e.g. `NUM_GPUS=2 bash run_walkerall.sh`.
#
# For pinning to specific physical GPU indices instead (e.g. testing on a
# shared local workstation), set GPU_LIST_ENV="1 3" to bypass NUM_GPUS.
NUM_GPUS="${NUM_GPUS:-4}"

if [[ -n "${GPU_LIST_ENV:-}" ]]; then
    read -ra GPU_LIST <<< "${GPU_LIST_ENV}"
else
    GPU_LIST=()
    for ((i = 0; i < NUM_GPUS; i++)); do
        GPU_LIST+=("${i}")
    done
fi

# NOTE: with only one seed here, only one GPU worker ever does anything --
# the rest of GPU_LIST sits idle (see run_gpu_worker: each worker only picks
# up every NUM_GPUS-th seed). Add more seeds to actually use more GPUs.
SEEDS=(1 2 3 4)
RUN_MODE="${1:-both}"
SIM1_EXP_NAME="mpo_walker"
SIM1_TOTAL_TIMESTEPS="2_000_000"
GLOBAL_TOTAL_TIMESTEPS="5_000_000"

RUNS_DIR="/n/holylfs05/LABS/hankyang_lab/Lab/serenaliu/walker2d_v5/EMPO"

MODEL1_PATH="/n/home06/serenaliu/open-ant/sim/assets/walker2d_v5.xml"
MODEL2_PATH="/n/home06/serenaliu/open-ant/sim/assets/walker2d_sim2.xml"

CRITIC_TYPE="scalar"
ENSEMBLE=3

CONT_EXP_NAME="continual_mpo_walker"



run_seed_pipeline() {
    local seed="$1"
    local physical_gpu="$2"
    local sim1_run_dir=""
    local weights_path=""

    # -------------------------------------------------------------------------
    # Sim1
    # -------------------------------------------------------------------------
    if [[ "${RUN_MODE}" != "sim2" ]]; then
        echo
        echo "Seed ${seed}: starting Sim1 on physical GPU ${physical_gpu}"

        local marker
        marker="$(mktemp)"
        touch "${marker}"

        CUDA_VISIBLE_DEVICES="${physical_gpu}" python3 "${SCRIPT}" \
            --env_id Walker2d-v5 \
            --render_mode rgb_array \
            --exp_name "${SIM1_EXP_NAME}" \
            --total_timesteps "${SIM1_TOTAL_TIMESTEPS}" \
            --seed "${seed}" \
            --runs_directory "${RUNS_DIR}" \
            --cuda \
            --model_path "${MODEL1_PATH}" \
            --critic_type "${CRITIC_TYPE}" \
            --ensemble "${ENSEMBLE}"

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
        if [[ "${RUN_MODE}" == "sim1" ]]; then
            return
        fi
    fi

    # -------------------------------------------------------------------------
    # Find existing Sim1 run for Sim2-only mode
    # -------------------------------------------------------------------------
    if [[ "${RUN_MODE}" == "sim2" ]]; then
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
        --ensemble "${ENSEMBLE}"
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
