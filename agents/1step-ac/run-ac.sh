#!/bin/bash

SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do

    echo "Running seed $SEED: one-step actor-critic on CartPole..."

    python3 onestep_ac_v2.py \
        --exp_name one_step_ac \
        --run_dir runs_cartpole/experiment16 \
        --total_timesteps 40000 \
        --seed $SEED \
        --actor_lr 0.0001 \
        --critic_lr 0.0001 \
        --gamma 0.99 \
        --capture_video \
        --save_video_every_n_episodes 20 \
        --flush_log_every_n_episodes 1

    echo "Done with seed $SEED"
done

echo "All seeds complete!"
