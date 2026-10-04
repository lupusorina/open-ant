#!/bin/bash

# Learn in simulation.
if [ "$1" == "sim" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_test \
        --exp_name trial_1 \
        --num_envs 1
fi

# Learn on hardware.
if [ "$1" == "hw" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --runs_directory runs_hw_new_refactored_code \
        --exp_name trial_1 \
        --seed 1 \
        # --eval True
fi

if [ "$1" == "minipi" ]; then
    MUJOCO_GL=egl python3 sac_cleanrl.py \
    --env_id MiniPiWalk-v0 \
    --model_path ../../sim/assets/hightorque/scene.xml \
    --exp_name sac_minipi_upright \
    --seed 1 \
    --total_timesteps 1000000 \
    --num_envs 1 \
    --cuda \
    --gamma 0.99 \
    --learning_starts 5000 \
    --capture_video \
    --capture_video_steps 1000 \
    --capture_video_every 10000 \
    --save_every_n_steps 50000
fi

if [ "$1" == "pusher" ]; then
    MUJOCO_GL=egl python3 sac_cleanrl.py \
    --env_id Pusher-v5 \
    --model_path ../../sim/assets/pusher/pusher.xml \
    --exp_name sac_pusher \
    --seed 1 \
    --total_timesteps 1000000 \
    --num_envs 1 \
    --cuda \
    --gamma 0.99 \
    --learning_starts 5000 \
    --capture_video \
    --capture_video_steps 1000 \
    --capture_video_every 10000 \
    --save_every_n_steps 50000
fi

if [ "$1" == "reacher" ]; then
    MUJOCO_GL=egl python3 sac_cleanrl.py \
    --env_id Reacher-v5 \
    --model_path ../../sim/assets/reacher/reacher.xml \
    --exp_name sac_reacher \
    --seed 1 \
    --total_timesteps 1000000 \
    --num_envs 1 \
    --cuda \
    --gamma 0.99 \
    --learning_starts 5000 \
    --capture_video \
    --capture_video_steps 1000 \
    --capture_video_every 10000 \
    --save_every_n_steps 50000
fi

if [ "$1" == "swimmer" ]; then
    MUJOCO_GL=egl python3 sac_cleanrl.py \
    --env_id Swimmer-v5 \
    --model_path ../../sim/assets/swimmer/swimmer.xml \
    --exp_name sac_swimmer \
    --seed 1 \
    --total_timesteps 1000000 \
    --num_envs 1 \
    --cuda \
    --gamma 0.99 \
    --learning_starts 5000 \
    --capture_video \
    --capture_video_steps 1000 \
    --capture_video_every 10000 \
    --save_every_n_steps 50000
fi