#!/bin/bash

SEEDS=(0 1 2 3 4 5)

for SEED in "${SEEDS[@]}"; do

    echo "Running seed $SEED: one-step actor-critic on CartPole..."

    python3 idbd3.py \
        --exp_name one_step_idbd \
        --run_dir runs_cartpole/idbd-6 \
        --num_episodes 5000 \
        --seed $SEED \
        --actor_lr 0.0001 \
        --critic_lr 0.0001 \
        --actor_meta_lr 1e-6 \
        --critic_meta_lr 1e-6 \
        --min_lr 1e-6 \
        --max_lr 1e-2 \
        --gamma 0.99 \
        --capture_video \
        --save_video_every_n_episodes 20 \
        --flush_log_every_n_episodes 40 \
        --no-use_layer_norm \

    echo "Done with seed $SEED"
done

echo "All seeds complete!"
