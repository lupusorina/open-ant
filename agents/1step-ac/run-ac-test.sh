#!/bin/bash

SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do

    echo "Running seed $SEED: one-step actor-critic on CartPole..."

    python3 test_onestep_ac.py \
        --exp_name one_step_ac \
        --run_dir runs_cartpole/experiment14 \
        --num_episodes 5000 \
        --seed $SEED \
        --actor_lr 0.0001 \
        --critic_lr 0.0001 \
        --gamma 0.99 \
        --capture_video \
        --save_video_every_n_episodes 40 \
        --no-use_layer_norm


    echo "Done with seed $SEED"
done

echo "All seeds complete!"
