#!/usr/bin/env bash
#SBATCH --job-name=walkerall
#SBATCH --account=hankyang_lab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage:
#   sbatch submit_walkerall.slurm [both|sim1|sim2]
#
# Requests 4 dedicated A100s (80GB) on FASRC's `gpu` partition, which is
# entirely nvidia_a100-sxm4-80gb nodes and (unlike gpu_requeue) is not
# preemptible.
#
# NOTE: run_walkerall.sh's SEEDS list needs one seed per GPU worker to keep
# all 4 GPUs busy. If you change the GPU count, change it in both places:
# the --gres line above and NUM_GPUS below.

set -euo pipefail

# $0 inside an sbatch job points at Slurm's copy of this script in its spool
# dir (which you can't write to), not the original file -- use
# SLURM_SUBMIT_DIR (the directory `sbatch` was run from) instead.
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /n/home06/serenaliu/open-ant/ant_env/bin/activate

export NUM_GPUS=4

bash run_walkerall.sh "${1:-both}"
