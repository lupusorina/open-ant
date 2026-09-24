#!/usr/bin/env bash
# MPO Sim2-only (continual learning) sweep over generated Walker2d XML variants, in parallel across GPUs.
# Every variant warm-starts from the SAME existing Sim1 checkpoint (<sim1 run>/weights_and_args).
#
# Usage: bash run_walker_sweep.sh            (warm-start from Sim1 checkpoint)
#        bash run_walker_sweep.sh --scratch  (train each variant from scratch, no checkpoint)
#        bash run_walker_sweep.sh --resume   (continue interrupted runs in place: same run dir/CSVs,
#                                             from the latest checkpoint up to GLOBAL_TOTAL_TIMESTEPS)
#        --scratch and --resume combine (in either order) to resume scratch_sweep_* runs.
#        Resume uses the newest <name>_*_seed_<SEED> dir per variant; set SEEDS/VARIANTS to pick runs.
# (edit GPU_LIST / PER_GPU / VARIANTS / SEEDS below)
#
# Variant XMLs come from ../sac/make_walker_variants.py (regenerated at start). Runs are named
# sweep_<variant> (or scratch_sweep_<variant> with --scratch), so results land in
# ${RUNS_DIR}/<name>_<date>_seed_<seed>.

set -euo pipefail
cd "$(dirname "$0")"

SCRATCH=0
RESUME=0
for a in "$@"; do
    [[ "${a}" == "--scratch" ]] && SCRATCH=1
    [[ "${a}" == "--resume" ]] && RESUME=1
done

PY="${PY:-python3}"
SCRIPT="mpo_acme.py"

# ---- edit these ----
GPU_LIST=(2)                     # physical GPU ids to use, e.g. (0 1 2 3)
PER_GPU=1                       # concurrent runs per GPU
SEEDS=(1)                      # one job per (variant, seed); spread over the GPU workers
VARIANTS=(dens2_foot0.5)                      # empty = all in manifest; or e.g. (dens2 foot0.2 all_hard)
SIM1_RUN_DIR=""                  # empty = auto-detect latest sim1 run PER SEED; or pin one run dir (single seed only)
# --------------------

SIM1_EXP_NAME="mpo_walker"
GLOBAL_TOTAL_TIMESTEPS="1_500_000"
RUNS_DIR="/data2/serenaliu_data/1walker_moreaggressive/2dmpo_walker"

CRITIC_TYPE="categorical"
ENSEMBLE=1

GEN_SCRIPT="../sac/make_walker_variants.py"
GEN_DIR="$(cd ../../sim/assets && pwd)/generated"

"${PY}" "${GEN_SCRIPT}" > /dev/null

if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    mapfile -t VARIANTS < "${GEN_DIR}/manifest.txt"
fi

if [[ "${SCRATCH}" -eq 1 ]]; then
    NAME_PREFIX="scratch_sweep"
    echo "Mode: from scratch (no checkpoint)"
else
    NAME_PREFIX="sweep"
fi
[[ "${RESUME}" -eq 1 ]] && echo "Mode: resume in place"

# Sim1 checkpoint dir for a seed (warm-start mode only)
sim1_weights() {
    local seed="$1" dir="${SIM1_RUN_DIR}"
    [[ -n "${dir}" ]] || dir="$(
        find "${RUNS_DIR}" -maxdepth 1 -type d -name "${SIM1_EXP_NAME}_*_seed_${seed}" \
            -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
    )"
    [[ -d "${dir}/weights_and_args" ]] || { echo "ERROR: missing sim1 weights '${dir}/weights_and_args' for seed ${seed}" >&2; return 1; }
    echo "${dir}/weights_and_args"
}

# Fail fast if any seed lacks a sim1 checkpoint
if [[ "${SCRATCH}" -eq 0 && "${RESUME}" -eq 0 ]]; then
    for seed in "${SEEDS[@]}"; do echo "Sim1 checkpoint (seed ${seed}): $(sim1_weights "${seed}")" || exit 1; done
fi
echo "Variants (${#VARIANTS[@]}): ${VARIANTS[*]}"
echo "Seeds: ${SEEDS[*]}"

run_variant() {
    local variant="$1" seed="$2" gpu="$3"
    local xml="${GEN_DIR}/walker2d_${variant}.xml"
    [[ -f "${xml}" ]] || { echo "ERROR: missing ${xml}" >&2; return 1; }
    local resume_args=()
    if [[ "${RESUME}" -eq 1 ]]; then
        local run_dir step
        run_dir="$(find "${RUNS_DIR}" -maxdepth 1 -type d -name "${NAME_PREFIX}_${variant}_*_seed_${seed}" \
            -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
        [[ -d "${run_dir}/weights_and_args" ]] || { echo "ERROR: no run to resume for ${variant} seed ${seed}" >&2; return 1; }
        step="$(find "${run_dir}/weights_and_args" -maxdepth 1 -name 'checkpoint_*.pth' \
            | sed -E 's/.*checkpoint_([0-9]+)\.pth/\1/' | sort -n | tail -n 1)"
        [[ -n "${step}" ]] || { echo "ERROR: no checkpoint in ${run_dir}" >&2; return 1; }
        if (( step >= ${GLOBAL_TOTAL_TIMESTEPS//_/} )); then
            echo "[GPU ${gpu}] ${variant}: already at ${step}, skipping"; return 0
        fi
        echo "[GPU ${gpu}] ${variant}: resuming ${run_dir} from step ${step}"
        resume_args=(--resume_in_place --weights_path "${run_dir}/weights_and_args")
    else
        if [[ "${SCRATCH}" -eq 1 ]]; then
            resume_args=()
        else
            # Continual learning: load sim1 networks/optimizers/duals but NOT the sim1 replay buffer.
            # The .py loads replay_buffer.npz only if it exists in --weights_path, so point it at a
            # folder that links to the latest sim1 checkpoint and nothing else.
            local src ckpt_step nobuf
            src="$(sim1_weights "${seed}")"
            ckpt_step="$(find "${src}" -maxdepth 1 -name 'checkpoint_*.pth' \
                | sed -E 's/.*checkpoint_([0-9]+)\.pth/\1/' | sort -n | tail -n 1)"
            nobuf="${RUNS_DIR}/.sim1_nobuffer/${variant}_seed_${seed}"
            rm -rf "${nobuf}" && mkdir -p "${nobuf}"
            ln -s "${src}/checkpoint_${ckpt_step}.pth" "${nobuf}/"
            echo "[GPU ${gpu}] ${variant}: warm start from ${src}/checkpoint_${ckpt_step}.pth (no sim1 replay buffer)"
            resume_args=(--weights_path "${nobuf}")
        fi
    fi
    echo "[GPU ${gpu}] ${variant}: ${xml}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${SCRIPT}" \
        --env_id Walker2d-v5 \
        --render_mode rgb_array \
        --exp_name "${NAME_PREFIX}_${variant}" \
        --total_timesteps "${GLOBAL_TOTAL_TIMESTEPS}" \
        --seed "${seed}" \
        --runs_directory "${RUNS_DIR}" \
        "${resume_args[@]}" \
        --model_path "${xml}" \
        --cuda \
        --critic_type "${CRITIC_TYPE}" \
        --ensemble "${ENSEMBLE}" \
        --gamma 0.99 \
        --dual_lr 0.005 \
        --policy_init_scale 0.5 \
        --samples_per_insert 192 \
        --vmin -650 \
        --vmax 650 \
        >> "${RUNS_DIR}/${NAME_PREFIX}_${variant}_seed${seed}.log" 2>&1
    echo "[GPU ${gpu}] ${variant}: done"
}

# Jobs = variant x seed (seed varies fastest). Worker w takes jobs w, w+N, w+2N, ...
# so with 2 GPUs and 2 seeds, each GPU gets its own seed.
JOBS=()
for v in "${VARIANTS[@]}"; do for sd in "${SEEDS[@]}"; do JOBS+=("${v}:${sd}"); done; done
NW=$(( ${#GPU_LIST[@]} * PER_GPU ))
pids=()
for ((w=0; w<NW && w<${#JOBS[@]}; w++)); do
    (
        gpu="${GPU_LIST[$((w % ${#GPU_LIST[@]}))]}"
        for ((i=w; i<${#JOBS[@]}; i+=NW)); do
            job="${JOBS[$i]}"
            run_variant "${job%:*}" "${job##*:}" "${gpu}" || echo "FAILED: ${job}" >&2
        done
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
[[ "${failed}" -eq 0 ]] || { echo "Some workers failed." >&2; exit 1; }
echo "All sweep runs finished. Logs: ${RUNS_DIR}/${NAME_PREFIX}_*_seed*.log"
