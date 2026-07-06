
#!/bin/bash
 
SEEDS=(0 1 2)
 
for SEED in "${SEEDS[@]}"; do
 
    echo "Running seed $SEED: one-step actor-critic on Ant..."
 
    python3 onestep_ac_ant.py \
        --exp_name one_step_ac_ant \
        --run_dir runs_ant/experiment5 \
        --total_timesteps 40000 \
        --seed $SEED \
        --actor_lr 3e-4 \
        --critic_lr 1e-3 \
        --gamma 0.99 \
        --task_type back_and_forth \
        --radius_back_and_forth 0.3 \
        --origin_back_and_forth 0.75 -0.3 \
        --dt 0.12 \
        --reward_scale 1.0 \
        --log_every_n_steps 4000 \
        --render_mode rgb_array \
        --save_video_every_n_steps 4000 \
        --capture_video
 
    echo "Done with seed $SEED"
done
 
echo "All seeds complete!"
