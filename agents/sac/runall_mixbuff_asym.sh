#!/bin/bash

# ← Edit this list to whatever seeds you want
SEEDS=(0 1 2 3 4 5 6 7 8 9 10)

for SEED in "${SEEDS[@]}"; do
    echo "========================================="
    echo "Running seed $SEED: sim training..."
    echo "========================================="

    python3 mixbuff_sac.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs/sac_mixbuff_asym \
        --exp_name sac_trick_sim1 \
        --num_envs 1 \
        --total_timesteps 40000 \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id.xml \
        --dt 0.12 \
        --seed $SEED \
        --cuda

    SIM1_DIR=$(ls -td runs/sac_mixbuff_asym/sac_trick_sim1_*_seed_${SEED} | grep -v continual | head -1)
    echo "Sim1 run folder: $SIM1_DIR"

    echo "========================================="
    echo "Running seed $SEED: warm start continual learning..."
    echo "========================================="

    python3 mixbuff_sac.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs/sac_mixbuff_asym \
        --exp_name sac_trick_sim2 \
        --num_envs 1 \
        --dt 0.12 \
        --weights_path $SIM1_DIR \
        --total_timesteps 150000 \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
        --policy_frequency 20 \
        --policy_lr 1e-5 \
        --q_lr 3e-4 \
        --seed $SEED \
        --asymmetric_updates \
        --offline_buffer_path $SIM1_DIR/replay_buffer \
        --cuda

    echo "Done with seed $SEED"
done

echo "All seeds complete!"
