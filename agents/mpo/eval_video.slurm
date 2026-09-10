#!/bin/bash
#SBATCH --job-name=eval_video
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=eval_video_%j.log

source /n/home06/serenaliu/open-ant/ant_env/bin/activate
cd /n/home06/serenaliu/open-ant/agents/mpo
python3 eval_videos_serna.py
