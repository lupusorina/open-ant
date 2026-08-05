#!/bin/bash
set -euo pipefail

# Usage:
#   bash runall.sh sim
#   bash runall.sh sim_continual_learning
#   bash runall.sh sim_then_continual
#   bash runall.sh hw

# Seeds run in this exact order.
SEEDS=(10 1 2 3 8 7 6 5 4 9 0)

# The ensemble agent is now a configuration of the merged agent.
SCRIPT="mpo_acme.py"
ENSEMBLE=3

# Sim1 and Sim2 both save under this same parent directory.
RUNS_DIR="runs/ensemble_spi1536_gamma92"

# Sim1 output folders:
#   runs-ant/mpo_YYYYMMDD-HHMMSS_seed_3
SIM_EXP_NAME="mpo"

# Sim2 output folders:
#   runs-ant/continuous_mpo_YYYYMMDD-HHMMSS_seed_3
CONT_EXP_NAME="continuous_mpo"

CAPTURE_VIDEO=0



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
        --ensemble "${ENSEMBLE}" \
        --render_mode rgb_array \
        --total_timesteps 40_000 \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --runs_directory "${RUNS_DIR}" \
        --exp_name "${SIM_EXP_NAME}" \
        --samples_per_insert 1536 \
        --seed "${seed}" \
        --gamma 0.92 \
        --cuda \
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

    local weights_path="${sim_run_dir}/weights_and_args"

    if [ ! -d "${weights_path}" ]; then
        echo "ERROR: Found Sim1 run dir, but no weights_and_args folder:"
        echo "  ${weights_path}"
        exit 1
    fi

    echo "Sim1 complete for seed ${seed}."
    echo "  ${sim_run_dir}"
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
        --ensemble "${ENSEMBLE}" \
        --render_mode rgb_array \
        --total_timesteps 120_000 \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --runs_directory "${RUNS_DIR}" \
        --samples_per_insert 1536 \
        --seed "${seed}" \
        --cuda \
        --exp_name "${CONT_EXP_NAME}" \
        --weights_path "${weights_path}" \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
        --gamma 0.92 \
        --cuda \
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
        --ensemble "${ENSEMBLE}" \
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
    echo "Usage: bash runall.sh {sim|sim_continual_learning|sim_then_continual|hw}"
    exit 1
fi
