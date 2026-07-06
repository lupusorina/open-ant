#!/bin/bash
set -euo pipefail

# Usage:
#   bash run-mpo-ant-continual.sh sim
#   bash run-mpo-ant-continual.sh sim_continual_learning
#   bash run-mpo-ant-continual.sh sim_then_continual
#   bash run-mpo-ant-continual.sh hw

SEEDS=(1 2 0 4 5 6 7 8 10)

# Change this if your W&B-modified MPO file has a different name.
SCRIPT="mpo.py"

# Keep exact parent directory name from your manual script.
# Note: this preserves your spelling: "continous", not "continuous".
RUNS_DIR="runs_continous_learning"

# Keep exact exp names from your manual script.
SIM_EXP_NAME="retrace"
CONT_EXP_NAME="retrace_continual_learning"

run_sim () {
    local seed="$1"

    echo "=========================================="
    echo "Running Sim1 for seed ${seed}"
    echo "Output parent dir: ${RUNS_DIR}"
    echo "Sim1 exp name: ${SIM_EXP_NAME}"
    echo "=========================================="

    local marker
    marker=$(mktemp)
    touch "${marker}"

    python3 "${SCRIPT}" \
        --render_mode rgb_array \
        --total_timesteps 40000 \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --runs_directory "${RUNS_DIR}" \
        --exp_name "${SIM_EXP_NAME}" \
        --utd_ratio 3 \
        --ensemble 3 \
        --decouple_q_learning \
        --policy_learning_starts 2000 \
        --td_horizon 3 \
        --log_every_n_steps 100 \
        --save_every_n_steps 4000 \
        --seed "${seed}" \
        --cuda \
        --track_wandb \
        --wandb_project mpo-ant-continual \
        --wandb_run_name "sim1_seed_${seed}_ensemble3" \
        --wandb_group "mpo_ensemble3" \
        --wandb_mode online

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
        echo "ERROR: Sim1 finished, but I could not find the new run dir for seed ${seed}."
        echo "Expected something like:"
        echo "  ${RUNS_DIR}/${SIM_EXP_NAME}_*_seed_${seed}"
        exit 1
    fi

    local weights_path="${sim_run_dir}/weights_and_args"

    if [ ! -d "${weights_path}" ]; then
        echo "ERROR: Found Sim1 run dir, but no weights_and_args folder:"
        echo "  ${weights_path}"
        exit 1
    fi

    echo "Sim1 complete for seed ${seed}."
    echo "Sim1 run dir:"
    echo "  ${sim_run_dir}"
    echo "Sim1 weights:"
    echo "  ${weights_path}"

    # Final line is intentionally only the weights path.
    # sim_then_continual captures this exact path.
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
        echo "ERROR: Could not find any Sim1 run dir for seed ${seed}."
        echo "Expected something like:"
        echo "  ${RUNS_DIR}/${SIM_EXP_NAME}_*_seed_${seed}"
        echo
        echo "Important: this function intentionally only searches Sim1 dirs."
        echo "It will not search continual-learning / Sim2 dirs."
        exit 1
    fi

    local weights_path="${sim_run_dir}/weights_and_args"

    if [ ! -d "${weights_path}" ]; then
        echo "ERROR: Found Sim1 run dir, but no weights_and_args folder:"
        echo "  ${weights_path}"
        exit 1
    fi

    echo "${weights_path}"
}

run_continual () {
    local seed="$1"
    local weights_path="$2"

    echo "=========================================="
    echo "Running continual learning / Sim2 for seed ${seed}"
    echo "Output parent dir: ${RUNS_DIR}"
    echo "Continual exp name: ${CONT_EXP_NAME}"
    echo "Loading Sim1 weights from:"
    echo "  ${weights_path}"
    echo "=========================================="

    python3 "${SCRIPT}" \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --total_timesteps 120000 \
        --dt 0.12 \
        --runs_directory "${RUNS_DIR}" \
        --exp_name "${CONT_EXP_NAME}" \
        --utd_ratio 3 \
        --ensemble 3 \
        --decouple_q_learning \
        --policy_learning_starts 2000 \
        --td_horizon 3 \
        --log_every_n_steps 100 \
        --save_every_n_steps 4000 \
        --weights_path "${weights_path}" \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
        --cuda \
        --seed "${seed}" \
        --track_wandb \
        --wandb_project mpo-ant-continual \
        --wandb_run_name "sim2_seed_${seed}_ensemble3" \
        --wandb_group "mpo_ensemble3" \
        --wandb_mode online

    echo "Continual learning complete for seed ${seed}."
}

if [ "$1" == "sim" ]; then
    for seed in "${SEEDS[@]}"; do
        run_sim "${seed}"
    done

elif [ "$1" == "sim_continual_learning" ]; then
    for seed in "${SEEDS[@]}"; do
        weights_path=$(find_latest_sim_weights_for_seed "${seed}")
        run_continual "${seed}" "${weights_path}"
    done

elif [ "$1" == "sim_then_continual" ]; then
    for seed in "${SEEDS[@]}"; do
        echo "##########################################"
        echo "Starting seed ${seed}: Sim1 then Sim2/continual"
        echo "##########################################"

        sim_output=$(run_sim "${seed}")
        echo "${sim_output}"

        weights_path=$(echo "${sim_output}" | tail -n 1)

        run_continual "${seed}" "${weights_path}"

        echo "Finished seed ${seed}."
    done

elif [ "$1" == "hw" ]; then
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
        --ensemble 1 \
        --decouple_q_learning \
        --policy_learning_starts 1000 \
        --td_horizon 3 \
        --log_every_n_steps 100 \
        --save_every_n_steps 4000 \
        --track_wandb \
        --wandb_project mpo-ant-continual \
        --wandb_run_name "hw_seed_1_singlecritic" \
        --wandb_group "mpo_singlecritic_hw" \
        --wandb_mode online

else
    echo "Usage: bash run-mpo-ant-continual.sh {sim|sim_continual_learning|sim_then_continual|hw}"
    exit 1
fi