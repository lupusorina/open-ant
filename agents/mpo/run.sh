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
