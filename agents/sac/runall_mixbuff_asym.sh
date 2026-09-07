#!/bin/bash

# Usage:
#   bash runall_mixbuff_asym.sh            # sim1 + continual for every seed
#   bash runall_mixbuff_asym.sh continual  # continual only, reusing existing sim1 checkpoints

# ← Edit this list to whatever seeds you want
SEEDS=(1 2 3 4 5 6 7 8 9 10)

RUNS_DIR="runs/sac_mixbuff_asym_freq2"

MODE="${1:-all}"

find_sim1_dir () {
    local seed="$1"
    ls -td "${RUNS_DIR}"/sac_sim1_*_seed_${seed} 2>/dev/null | grep -v continual | head -1
}

run_sim1 () {
    local seed="$1"

    python3 mixbuff_sac.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory "${RUNS_DIR}" \
        --exp_name sac_sim1 \
        --num_envs 1 \
        --total_timesteps 40000 \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id.xml \
        --dt 0.12 \
        --seed "${seed}" \
        --cuda
}

run_continual () {
    local seed="$1"
    local sim1_dir="$2"

    python3 mixbuff_sac.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory "${RUNS_DIR}" \
        --exp_name continuous_sac_freq2 \
        --num_envs 1 \
        --dt 0.12 \
        --weights_path "${sim1_dir}" \
        --total_timesteps 150000 \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
        --policy_frequency 2 \
        --policy_lr 1e-5 \
        --q_lr 3e-4 \
        --seed "${seed}" \
        --asymmetric_updates \
        --offline_buffer_path "${sim1_dir}/replay_buffer" \
        --cuda
}

for SEED in "${SEEDS[@]}"; do
    case "$MODE" in
        all)
            echo "========================================="
            echo "Running seed $SEED: sim training..."
            echo "========================================="
            run_sim1 "$SEED"

            SIM1_DIR=$(find_sim1_dir "$SEED")
            echo "Sim1 run folder: $SIM1_DIR"

            echo "========================================="
            echo "Running seed $SEED: continual learning..."
            echo "========================================="
            run_continual "$SEED" "$SIM1_DIR"
            ;;
        continual)
            SIM1_DIR=$(find_sim1_dir "$SEED")
            if [ -z "$SIM1_DIR" ]; then
                echo "ERROR: no sim1 run found for seed $SEED in ${RUNS_DIR} matching sac_sim1_*_seed_${SEED}"
                exit 1
            fi
            echo "Sim1 run folder: $SIM1_DIR"

            echo "========================================="
            echo "Running seed $SEED: continual learning..."
            echo "========================================="
            run_continual "$SEED" "$SIM1_DIR"
            ;;
        *)
            echo "Usage: bash runall_mixbuff_asym.sh {all|continual}"
            exit 1
            ;;
    esac

    echo "Done with seed $SEED"
done

echo "All seeds complete!"
