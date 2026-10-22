SCRIPT="mpo_acme.py"

# Learn on hardware.
if [ "$1" == "hw" ]; then
    python3 "${SCRIPT}" \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --runs_directory runs_hw_new_refactored_code \
        --exp_name trial_1 \
        --seed 1 \
        --cuda
        # --eval True
fi

if [ "$1" == "minipi" ]; then
    MUJOCO_GL=egl python3 mpo_acme.py \
    --env_id MiniPiWalk-v0 \
    --model_path ../../sim/assets/hightorque/scene.xml \
    --exp_name mpo_minipi_upright \
    --runs_directory runs/mpo_minipi \
    --seed 1 \
    --total_timesteps 1_000_000 \
    --cuda \
    --critic_type scalar \
    --ensemble 1 \
    --gamma 0.99 \
    --dual_lr 0.005 \
    --policy_init_scale 0.5 \
    --samples_per_insert 256 \
    --save_every_n_steps 50000 \
    --log_every_n_steps 10000 \
    --capture_video \
    --capture_video_steps 1000 \
    --capture_video_every 50000 \
    --render_mode rgb_array
fi

if [ "$1" == "pusher" ]; then
    MUJOCO_GL=egl python3 mpo_acme.py \
    --env_id Pusher-v5 \
    --model_path ../../sim/assets/pusher/pusher.xml \
    --exp_name mpo_pusher \
    --runs_directory runs/mpo_pusher \
    --seed 1 \
    --total_timesteps 1_000_000 \
    --cuda \
    --critic_type scalar \
    --ensemble 1 \
    --gamma 0.99 \
    --dual_lr 0.005 \
    --policy_init_scale 0.5 \
    --samples_per_insert 256 \
    --save_every_n_steps 50000 \
    --log_every_n_steps 10000 \
    --capture_video \
    --capture_video_steps 1000 \
    --capture_video_every 50000 \
    --render_mode rgb_array
fi

if [ "$1" == "reacher" ]; then
    MUJOCO_GL=egl python3 mpo_acme.py \
    --env_id Reacher-v5 \
    --model_path ../../sim/assets/reacher/reacher.xml \
    --exp_name mpo_reacher \
    --runs_directory runs/mpo_reacher \
    --seed 1 \
    --total_timesteps 1_000_000 \
    --cuda \
    --critic_type scalar \
    --ensemble 1 \
    --gamma 0.99 \
    --dual_lr 0.005 \
    --policy_init_scale 0.5 \
    --samples_per_insert 256 \
    --save_every_n_steps 50000 \
    --log_every_n_steps 10000 \
    --capture_video \
    --capture_video_steps 1000 \
    --capture_video_every 50000 \
    --render_mode rgb_array
fi

if [ "$1" == "swimmer" ]; then
    MUJOCO_GL=egl python3 mpo_acme.py \
    --env_id Swimmer-v5 \
    --model_path ../../sim/assets/swimmer/swimmer.xml \
    --exp_name mpo_swimmer \
    --runs_directory runs/mpo_swimmer \
    --seed 1 \
    --total_timesteps 1_000_000 \
    --cuda \
    --critic_type scalar \
    --ensemble 1 \
    --gamma 0.99 \
    --dual_lr 0.005 \
    --policy_init_scale 0.5 \
    --samples_per_insert 256 \
    --save_every_n_steps 50000 \
    --log_every_n_steps 10000 \
    --capture_video \
    --capture_video_steps 1000 \
    --capture_video_every 50000 \
    --render_mode rgb_array
fi