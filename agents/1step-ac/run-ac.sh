#!/bin/bash

SEEDS=(0 1 2 3 4 5)

for SEED in "${SEEDS[@]}"; do

    echo "Running seed $SEED: one-step actor-critic on CartPole..."

    python3 onestep_ac.py \
        --exp_name one_step_ac \
        --run_dir runs_cartpole/experiment5 \
        --num_episodes 1000 \
        --seed $SEED \
        --actor_lr 6.8e-05 \
        --critic_lr 0.0117 \
        --gamma 0.99 \
        --use_layer_norm False 

    echo "Done with seed $SEED"
done

echo "All seeds complete!"
