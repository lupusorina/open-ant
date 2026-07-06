#!/bin/bash
set -euo pipefail

# Usage:
#   bash run-mpo-ant-warmstart-continual.sh sim
#   bash run-mpo-ant-warmstart-continual.sh sim_continual_learning
#   bash run-mpo-ant-warmstart-continual.sh sim_then_continual
#   bash run-mpo-ant-warmstart-continual.sh hw
#
# IMPORTANT:
#   --warm_start is used ONLY for continual learning / Sim2.
#   Do NOT use --warm_start for Sim1 from scratch.

if [ "$#" -lt 1 ]; then
    echo "Usage: bash $0 {sim|sim_continual_learning|sim_then_continual|hw}"
    exit 1
fi

MODE="$1"

# Seeds run in this exact order.
SEEDS=(9 3 2 1 0 4 5 6 7 10)

SCRIPT="mpo.py"

# Keep exact parent directory name from your manual script.
# Note: this preserves your spelling: "continous", not "continuous".
RUNS_DIR="runs_continous_learning"

# Exp names.
SIM_EXP_NAME="retrace"
CONT_EXP_NAME="retrace_continual_learning_warmstart"

# Shared settings.
DT="0.12"
SIM_TOTAL_TIMESTEPS="40000"
CONT_TOTAL_TIMESTEPS="120000"

LEARNING_STARTS="2000"
POLICY_LEARNING_STARTS="2000"

TD_HORIZON="3"
UTD_RATIO="3"
ENSEMBLE="3"

SIM_MODEL_PATH="../../sim/assets/ant_with_camera_after_sys_id.xml"
CONT_MODEL_PATH="../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml"


check_weights_path () {
    local weights_path="$1"

    if [ ! -d "${weights_path}" ]; then
        echo "ERROR: weights_path does not exist:" >&2
        echo "  ${weights_path}" >&2
        exit 1
    fi

    local n_ckpts
    n_ckpts=$(find "${weights_path}" -maxdepth 1 -type f -name "checkpoint_*.pth" | wc -l)

    if [ "${n_ckpts}" -eq 0 ]; then
        echo "ERROR: weights_path exists but has no checkpoint_*.pth files:" >&2
        echo "  ${weights_path}" >&2
        exit 1
    fi
}


run_sim () {
    local seed="$1"

    echo "==========================================" >&2
    echo "Running Sim1 for seed ${seed}" >&2
    echo "Output parent dir: ${RUNS_DIR}" >&2
    echo "Sim1 exp name: ${SIM_EXP_NAME}" >&2
    echo "NO --warm_start here. This is normal Sim1 from scratch." >&2
    echo "==========================================" >&2

    mkdir -p "${RUNS_DIR}"

    local marker
    marker=$(mktemp)
    touch "${marker}"

    python3 "${SCRIPT}" \
        --render_mode rgb_array \
        --total_timesteps "${SIM_TOTAL_TIMESTEPS}" \
        --dt "${DT}" \
        --env_id SimEmbodiedAnt \
        --runs_directory "${RUNS_DIR}" \
        --exp_name "${SIM_EXP_NAME}" \
        --utd_ratio "${UTD_RATIO}" \
        --ensemble "${ENSEMBLE}" \
        --decouple_q_learning \
        --learning_starts "${LEARNING_STARTS}" \
        --policy_learning_starts "${POLICY_LEARNING_STARTS}" \
        --td_horizon "${TD_HORIZON}" \
        --model_path "${SIM_MODEL_PATH}" \
        --seed "${seed}" \
        --cuda 1>&2

    local sim_run_dir
    sim_run_dir=$(
        find "${RUNS_DIR}" \
            -maxdepth 1 \
            -type d \
            -name "${SIM_EXP_NAME}_*_seed_${seed}" \
            -newer "${marker}" \
            -printf "%T@ %p\n" 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )

    rm -f "${marker}"

    if [ -z "${sim_run_dir}" ]; then
        echo "ERROR: Sim1 finished, but I could not find the new run dir for seed ${seed}." >&2
        echo "Expected something like:" >&2
        echo "  ${RUNS_DIR}/${SIM_EXP_NAME}_*_seed_${seed}" >&2
        exit 1
    fi

    local weights_path="${sim_run_dir}/weights_and_args"
    check_weights_path "${weights_path}"

    echo "Sim1 complete for seed ${seed}." >&2
    echo "Sim1 run dir:" >&2
    echo "  ${sim_run_dir}" >&2
    echo "Sim1 weights:" >&2
    echo "  ${weights_path}" >&2

    # IMPORTANT:
    # This function prints ONLY the weights path to stdout.
    # All logs go to stderr so sim_then_continual can safely capture this.
    echo "${weights_path}"
}


find_latest_sim_weights_for_seed () {
    local seed="$1"

    local sim_run_dir
    sim_run_dir=$(
        find "${RUNS_DIR}" \
            -maxdepth 1 \
            -type d \
            -name "${SIM_EXP_NAME}_*_seed_${seed}" \
            -printf "%T@ %p\n" 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )

    if [ -z "${sim_run_dir}" ]; then
        echo "ERROR: Could not find any Sim1 run dir for seed ${seed}." >&2
        echo "Expected something like:" >&2
        echo "  ${RUNS_DIR}/${SIM_EXP_NAME}_*_seed_${seed}" >&2
        echo >&2
        echo "Important: this intentionally searches only Sim1 dirs." >&2
        echo "It will not search continual-learning / Sim2 dirs." >&2
        exit 1
    fi

    local weights_path="${sim_run_dir}/weights_and_args"
    check_weights_path "${weights_path}"

    echo "${weights_path}"
}


run_continual () {
    local seed="$1"
    local weights_path="$2"

    check_weights_path "${weights_path}"

    echo "=========================================="
    echo "Running WARM-START continual learning / Sim2 for seed ${seed}"
    echo "Output parent dir: ${RUNS_DIR}"
    echo "Continual exp name: ${CONT_EXP_NAME}"
    echo "Loading Sim1 actor/critic from:"
    echo "  ${weights_path}"
    echo
    echo "Warm-start behavior:"
    echo "  - loads Sim1 policy/critic"
    echo "  - skips Sim1 replay buffer"
    echo "  - resets global_step to 0"
    echo "  - collects fresh Sim2 buffer using pi0"
    echo "  - no updates until learning_starts=${LEARNING_STARTS}"
    echo "  - decouple_q_learning then gives extra Q-only warmup of ${POLICY_LEARNING_STARTS} steps"
    echo "=========================================="

    python3 "${SCRIPT}" \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --total_timesteps "${CONT_TOTAL_TIMESTEPS}" \
        --dt "${DT}" \
        --runs_directory "${RUNS_DIR}" \
        --exp_name "${CONT_EXP_NAME}" \
        --utd_ratio "${UTD_RATIO}" \
        --ensemble "${ENSEMBLE}" \
        --decouple_q_learning \
        --learning_starts "${LEARNING_STARTS}" \
        --policy_learning_starts "${POLICY_LEARNING_STARTS}" \
        --td_horizon "${TD_HORIZON}" \
        --weights_path "${weights_path}" \
        --model_path "${CONT_MODEL_PATH}" \
        --warm_start \
        --cuda \
        --seed "${seed}"

    echo "Warm-start continual learning complete for seed ${seed}."
}


run_hw () {
    python3 "${SCRIPT}" \
        --render_mode rgb_array \
        --dt 0.20 \
        --total_timesteps 60000 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --runs_directory runs_hw \
        --exp_name larger_dt \
        --seed 1 \
        --utd_ratio 3 \
        --ensemble 3 \
        --decouple_q_learning \
        --policy_learning_starts 1000 \
        --td_horizon 3
}


if [ "${MODE}" == "sim" ]; then
    for seed in "${SEEDS[@]}"; do
        run_sim "${seed}"
    done

elif [ "${MODE}" == "sim_continual_learning" ]; then
    for seed in "${SEEDS[@]}"; do
        weights_path=$(find_latest_sim_weights_for_seed "${seed}")
        run_continual "${seed}" "${weights_path}"
    done

elif [ "${MODE}" == "sim_then_continual" ]; then
    for seed in "${SEEDS[@]}"; do
        echo "##########################################"
        echo "Starting seed ${seed}: Sim1 then warm-start Sim2/continual"
        echo "##########################################"

        weights_path=$(run_sim "${seed}")
        run_continual "${seed}" "${weights_path}"

        echo "Finished seed ${seed}."
    done

elif [ "${MODE}" == "hw" ]; then
    run_hw

else
    echo "Usage: bash $0 {sim|sim_continual_learning|sim_then_continual|hw}"
    exit 1
fi

