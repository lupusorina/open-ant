#!/bin/bash

SEEDS=(0)

for SEED in "${SEEDS[@]}"; do

    echo "Running seed $SEED: one-step actor-critic on CartPole..."

    python3 onestep_ac_v2.py \
        --exp_name one_step_ac \
        --run_dir runs_cartpole/experiment5 \
        --num_episodes 5000 \
        --seed $SEED \
        --actor_lr 0.0001 \
        --critic_lr 0.0001 \
        --gamma 0.99 \
        --no-use_layer_norm \
        --capture_video \
        --save_video_every_n_episodes 20

    echo "Done with seed $SEED"
done

echo "All seeds complete!"
