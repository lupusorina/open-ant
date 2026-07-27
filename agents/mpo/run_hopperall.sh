#!/usr/bin/env bash
# Run independent Hopper Sim1 -> Sim2 pipelines.
# At most one pipeline runs on each selected GPU.

set -euo pipefail
cd "$(dirname "$0")"

SCRIPT="mpo_acme.py"

GPU_LIST=(2)

SEEDS=(10)

SIM1_EXP_NAME="mpo_hopper"
SIM1_TOTAL_TIMESTEPS="2_000_000"
SIM2_TOTAL_TIMESTEPS="5_000_000"

RUNS_DIR="/data2/serenaliu_data/mpo_hopper_newparam"

MODEL_PATH="/home/serenaliu/caltech_linc_home/open-ant/sim/assets/hopper_sim2.xml"

CONT_EXP_NAME="continual_mpo_hopper"



run_seed_pipeline() {
    local seed="$1"
    local physical_gpu="$2"

    echo
    echo "Seed ${seed}: starting Sim1 on physical GPU ${physical_gpu}"

    # Allows us to find the exact run directory created by this invocation,
    # instead of accidentally loading an older run for the same seed.
    local marker
    marker="$(mktemp)"
    touch "${marker}"
    #SIM 1
    CUDA_VISIBLE_DEVICES="${physical_gpu}" python3 "${SCRIPT}" \
        --env_id Hopper-v5 \
        --exp_name "${SIM1_EXP_NAME}" \
        --total_timesteps "${SIM1_TOTAL_TIMESTEPS}" \
        --seed "${seed}" \
        --runs_directory "${RUNS_DIR}" \
        --cuda \
        --policy_init_scale 0.7 \
        --samples_per_insert 1536 \
        --gamma 0.92 \
        --dual_lr 1e-3 \
        --batch_size 512 \

    local sim1_run_dir
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

    local weights_path="${sim1_run_dir}/weights_and_args"

    if [[ ! -d "${weights_path}" ]]; then
        echo "ERROR: Missing weights directory:" >&2
        echo "  ${weights_path}" >&2
        return 1
    fi

    if ! find "${weights_path}" \
        -maxdepth 1 \
        -type f \
        -name 'checkpoint_*.pth' \
        -print -quit |
        grep -q .
    then
        echo "ERROR: No checkpoint_*.pth found in:" >&2
        echo "  ${weights_path}" >&2
        return 1
    fi
    echo
    echo "Seed ${seed}: Sim1 finished."
    echo "Sim1 run: ${sim1_run_dir}"

    # -------------------------------------------------------------------------
    # Sim2
    # -------------------------------------------------------------------------

    echo
    echo "Seed ${seed}: immediately starting Sim2"
    echo "Physical GPU: ${physical_gpu}"
    echo "Sim2 output: ${RUNS_DIR}"
    echo "Sim2 model:  ${MODEL_PATH}"
    echo "Loading:     ${weights_path}"

    CUDA_VISIBLE_DEVICES="${physical_gpu}" python3 "${SCRIPT}" \
        --env_id Hopper-v5 \
        --exp_name "${CONT_EXP_NAME}" \
        --total_timesteps "${SIM2_TOTAL_TIMESTEPS}" \
        --seed "${seed}" \
        --runs_directory "${RUNS_DIR}" \
        --weights_path "${weights_path}" \
        --model_path "${MODEL_PATH}" \
        --cuda \
        --policy_init_scale 0.7 \
        --samples_per_insert 1536 \
        --gamma 0.92 \
        --dual_lr 1e-3 \
        --batch_size 512 \

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
echo "All Hopper Sim1 -> Sim2 pipelines finished."
echo "GPUs used: ${GPU_LIST[*]}"
echo "============================================================"
