#!/bin/bash
set -euo pipefail

# Usage:
#   bash run-dmpo-ant.sh sim
#   bash run-dmpo-ant.sh sim_continual_learning
#   bash run-dmpo-ant.sh sim_then_continual
#   bash run-dmpo-ant.sh hw

# Seeds run in this exact order.
SEEDS=(0 9 1 2 3 8 7 6 5 4 10)

SCRIPT="dmpo_ant.py"

# Sim1 and Sim2 both save under this same parent directory.
RUNS_DIR="runs-ant/exp5"

# Sim1 output folders:
#   runs-ant/dmpo_retrace_YYYYMMDD-HHMMSS_seed_3
SIM_EXP_NAME="dmpo"

# Sim2 output folders:
#   runs-ant/continuous_dmpo_retrace_YYYYMMDD-HHMMSS_seed_3
CONT_EXP_NAME="continuous_dmpo"

CAPTURE_VIDEO=1
WANDB_PROJECT="dmpo-ant-exp5"
WANDB_ENTITY=""          # leave empty unless you have a team/entity
WANDB_MODE="online"      # use "offline" if no internet on cluster
WANDB_WATCH=0            # keep 0; watching gradients can slow training a lot

wandb_flag () {
    local phase="$1"
    local seed="$2"

    local flags=(
        --track_wandb
        --wandb_project "${WANDB_PROJECT}"
        --wandb_group "exp3_seed_${seed}"
        --wandb_run_name "${phase}_seed_${seed}"
        --wandb_mode "${WANDB_MODE}"
        --log_every_n_steps 100
    )

    if [ -n "${WANDB_ENTITY}" ]; then
        flags+=(--wandb_entity "${WANDB_ENTITY}")
    fi

    if [ "${WANDB_WATCH}" -eq 1 ]; then
        flags+=(--wandb_watch)
    fi

    printf '%q ' "${flags[@]}"
}

video_flag () {
    if [ "${CAPTURE_VIDEO}" -eq 1 ]; then
        echo "--capture_video"
    fi
}

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
        --decouple_q_learning \
        --policy_learning_starts 2000 \
        --td_horizon 1 \
        --num_atoms 101 \
        --vmin -500 \
        --vmax 20 \
        --tau 0.005 \
        --critic_num_samples 32 \
        --max_grad_norm 1 \
        --policy_hidden_dim 256 \
        --policy_n_hidden_layers 2 \
        --critic_hidden_dim 512 \
        --critic_n_hidden_layers 3 \
        --seed "${seed}" \
        --cuda \
        $(wandb_flag "sim1" "${seed}") \
        $(video_flag)

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
        echo "It will not search continuous/sim2 dirs."
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
        --decouple_q_learning \
        --learning_starts 2000 \
        --policy_learning_starts 2000 \
        --td_horizon 1 \
        --tau 0.005 \
        --num_atoms 101 \
        --vmin -500 \
        --vmax 20 \
        --critic_num_samples 32 \
        --max_grad_norm 1 \
        --policy_hidden_dim 256 \
        --policy_n_hidden_layers 2 \
        --critic_hidden_dim 512 \
        --critic_n_hidden_layers 3 \
        --weights_path "${weights_path}" \
        --warm_start \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
        --seed "${seed}" \
        --cuda \
        $(wandb_flag "sim2_warmstart" "${seed}") \
        $(video_flag)

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
        --decouple_q_learning \
        --policy_learning_starts 1000 \
        --td_horizon 1 \
        --num_atoms 101 \
        --vmin -500 \
        --vmax 20 \
        --critic_num_samples 32 \
        --max_grad_norm 1 \
        --policy_hidden_dim 256 \
        --policy_n_hidden_layers 2 \
        --critic_hidden_dim 256 \
        --critic_n_hidden_layers 2 \
        --cuda \
        $(video_flag)

else
    echo "Usage: bash run-dmpo-ant.sh {sim|sim_continual_learning|sim_then_continual|hw}"
    exit 1
fi