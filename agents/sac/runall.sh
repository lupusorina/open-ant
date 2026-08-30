#!/bin/bash
set -euo pipefail

# Usage:
#   bash run.sh sim
#   bash run.sh sim_continual_learning
#   bash run.sh sim_then_continual
#   bash run.sh hw

SEEDS=(3)

SCRIPT="sac_perweightopt.py"
RUNS_DIR="runs/idbd_newweight_nolayernorm_resetalpha"

SIM1_XML="/home/seliu/open-ant/sim/assets/ant_with_camera_after_sys_id.xml"
SIM2_XML="/home/seliu/open-ant/sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml"
SIM1_STEPS=40000
SIM2_STEPS=2000000


# =========================
# SIM1
# =========================
run_sim1 () {
    SEED="$1"

    python3 "${SCRIPT}" \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --model_path "${SIM1_XML}" \
        --total_timesteps "${SIM1_STEPS}" \
        --dt 0.12 \
        --meta_gamma 0.99 \
        --runs_directory "${RUNS_DIR}" \
        --exp_name sac_sim1 \
        --num_envs 1 \
        --seed "${SEED}" \
        --cuda \
        --no-use_layer_norm
}


# =========================
# SIM2
# =========================
run_sim2 () {
    SEED="$1"
    WEIGHTS_PATH="$2"

    python3 "${SCRIPT}" \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --model_path "${SIM2_XML}" \
        --total_timesteps "$((SIM1_STEPS + SIM2_STEPS))" \
        --dt 0.12 \
        --meta_gamma 0.99 \
        --runs_directory "${RUNS_DIR}" \
        --exp_name continuous_sac \
        --num_envs 1 \
        --seed "${SEED}" \
        --weights_path "${WEIGHTS_PATH}" \
        --cuda \
        --load_extra_weights \
        --num_extra_weights 26 \
        --no-use_layer_norm \
        --reset_entropy_alpha
}


# =========================
# HARDWARE
# =========================
run_hw () {
    SEED="$1"

    python3 "${SCRIPT}" \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --runs_directory runs_hw_new_refactored_code \
        --exp_name trial_1 \
        --seed "${SEED}" \
        --cuda \
        --no-use_layer_norm 
}


# =========================
# FIND LATEST SIM1 RUN
# =========================
find_sim1_weights () {
    SEED="$1"

    find "${RUNS_DIR}" \
        -maxdepth 1 \
        -type d \
        -name "sac_sim1_*_seed_${SEED}" \
        | sort \
        | tail -n 1
}


# =========================
# MODES
# =========================
if [ "$1" == "sim" ]; then

    for SEED in "${SEEDS[@]}"; do
        echo "========================================"
        echo "Running Sim1 seed ${SEED}"
        echo "========================================"

        run_sim1 "${SEED}"
    done


elif [ "$1" == "sim_continual_learning" ]; then

    for SEED in "${SEEDS[@]}"; do

        WEIGHTS_PATH=$(find_sim1_weights "${SEED}")

        if [ -z "${WEIGHTS_PATH}" ]; then
            echo "ERROR: No Sim1 run found for seed ${SEED} in ${RUNS_DIR}"
            exit 1
        fi

        echo "========================================"
        echo "Running Sim2 seed ${SEED}"
        echo "Loading checkpoint from: ${WEIGHTS_PATH}"
        echo "========================================"

        run_sim2 "${SEED}" "${WEIGHTS_PATH}"
    done


elif [ "$1" == "sim_then_continual" ]; then

    for SEED in "${SEEDS[@]}"; do

        echo "========================================"
        echo "Running Sim1 seed ${SEED}"
        echo "========================================"

        run_sim1 "${SEED}"

        WEIGHTS_PATH=$(find_sim1_weights "${SEED}")

        if [ -z "${WEIGHTS_PATH}" ]; then
            echo "ERROR: No Sim1 run found for seed ${SEED} in ${RUNS_DIR}"
            exit 1
        fi

        echo "========================================"
        echo "Running Sim2 seed ${SEED}"
        echo "Loading checkpoint from: ${WEIGHTS_PATH}"
        echo "========================================"

        run_sim2 "${SEED}" "${WEIGHTS_PATH}"

    done


elif [ "$1" == "hw" ]; then

    for SEED in "${SEEDS[@]}"; do
        echo "========================================"
        echo "Running hardware seed ${SEED}"
        echo "========================================"

        run_hw "${SEED}"
    done


else
    echo "Usage: bash run.sh {sim|sim_continual_learning|sim_then_continual|hw}"
    exit 1
fi