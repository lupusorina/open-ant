#!/usr/bin/env bash
# MPO (scalar critic, no ensemble) on the Mini Pi+ Pro walking task, Sim1 -> Sim2 sweep.
#
#   Sim1: MiniPiWalk-v0 on ../../sim/assets/hightorque/scene.xml, one run per seed.
#   Sim2: for every (variant, seed), warm-start from that seed's finished Sim1
#         checkpoint (networks, optimizers, duals, and the Sim1 replay buffer)
#         and continue on a generated XML from
#         ../../sim/assets/hightorque/make_mini_pi_variants.py
#         (mass x(1+s), friction x(1-s), s = 20/40/60/80%).
#
# Every seed runs its own pipeline in parallel; once a seed's Sim1 is done, all
# of its Sim2 variants start in parallel. Concurrency is capped at PER_GPU runs
# per GPU (jobs queue for a free slot).
#
# By default every launch starts NEW runs (new timestamped dirs); existing runs
# are never touched. Pass --resume to pick up existing runs instead. With
# --resume, for each seed / (variant, seed) it detects existing runs:
#   Sim1 finished                      -> use it, go straight to Sim2
#   Sim1 still training (process alive) -> wait for it to finish, then Sim2
#   Sim1 stopped part-way               -> resume it in place, then Sim2
#   no Sim1                             -> train it from scratch, then Sim2
#   Sim2 finished / still training      -> skip
#   Sim2 stopped part-way               -> resume it in place
#
# Usage:
#   bash run_minipiall.sh                          # sim_then_continual (default): new Sim1, then Sim2
#   bash run_minipiall.sh sim                      # Sim1 only, no Sim2
#   bash run_minipiall.sh sim_continual_learning   # Sim2 only: never trains/resumes/waits on
#                                                  # Sim1; warm-starts from whatever the latest
#                                                  # Sim1 checkpoint is right now, finished or not
#   bash run_minipiall.sh [mode] --resume          # any mode: resume existing runs (see above)
#                                                  # instead of starting new ones
# Logs: ${RUNS_DIR}/logs/<exp_name>_seed<seed>.log

set -euo pipefail
cd "$(dirname "$0")"

RESUME=0
RUN_MODE="sim_then_continual"
for arg in "$@"; do
    case "${arg}" in
        --resume) RESUME=1 ;;
        *) RUN_MODE="${arg}" ;;
    esac
done
case "${RUN_MODE}" in
    sim|sim_continual_learning|sim_then_continual) ;;
    *)
        echo "Usage: bash run_minipiall.sh [sim|sim_continual_learning|sim_then_continual] [--resume]" >&2
        exit 1
        ;;
esac

# Headless MuJoCo rendering via EGL so --capture_video works without a display.
export MUJOCO_GL=egl

PY="${PY:-python3}"
SCRIPT="mpo_acme.py"

# ---- edit these ----
GPU_LIST=(2)                 # physical GPU ids (nvidia-smi numbering)
PER_GPU=4                    # max concurrent runs this script starts per GPU
SEEDS=(1 2)
VARIANTS=()                  # empty = all in manifest; e.g. (mini_pi_mass_fric_20 mini_pi_mass_fric_80)
# --------------------

RUNS_DIR="runs/mpo_minipi"

SIM1_EXP_NAME="mpo_minipi"
SIM1_TOTAL_TIMESTEPS="500_000"
SIM1_MODEL_PATH="../../sim/assets/hightorque/scene.xml"

SIM2_PREFIX="continual_minipi"   # Sim2 exp_name = ${SIM2_PREFIX}_<variant>
# Steps to train in Sim2 *on top of* Sim1. mpo_acme.py continues the step
# counter from the loaded checkpoint and stops at --total_timesteps, so Sim2's
# --total_timesteps is set to (Sim1 checkpoint step + SIM2_STEPS).
SIM2_STEPS="1_500_000"

CRITIC_TYPE="scalar"
ENSEMBLE=1

GEN_SCRIPT="../../sim/assets/hightorque/make_mini_pi_variants.py"
GEN_DIR="../../sim/assets/hightorque/generated"

POLL_SECONDS=600             # how often to re-check a Sim1 that is still training

COMMON_ARGS=(
    --env_id MiniPiWalk-v0
    --render_mode rgb_array
    --cuda
    --critic_type "${CRITIC_TYPE}"
    --ensemble "${ENSEMBLE}"
    --gamma 0.99
    --dual_lr 0.005
    --policy_init_scale 0.5
    --samples_per_insert 256
    --save_every_n_steps 50000
    --log_every_n_steps 10000
    --capture_video
    --capture_video_steps 1000
    --capture_video_every 50000
)


# Make CUDA ordinals match nvidia-smi indices.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# MuJoCo's EGL renderer ignores CUDA_VISIBLE_DEVICES and, by default, grabs EGL
# device 0. EGL device order does NOT match nvidia-smi order on this machine, so
# look up the EGL index whose CUDA device == the requested physical GPU.
egl_device_for_gpu() {
    env -u CUDA_VISIBLE_DEVICES PYOPENGL_PLATFORM=egl "${PY}" - "$1" <<'PY'
import ctypes, sys
from mujoco.egl import egl_ext as EGL
gpu = int(sys.argv[1])
gpa = ctypes.CDLL("libEGL.so.1").eglGetProcAddress
gpa.restype, gpa.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
                         ctypes.POINTER(ctypes.c_long))(gpa(b"eglQueryDeviceAttribEXT"))
for i, dev in enumerate(EGL.eglQueryDevicesEXT()):
    cuda_id = ctypes.c_long(-1)
    if query(ctypes.cast(dev, ctypes.c_void_p).value, 0x323A, ctypes.byref(cuda_id)) and cuda_id.value == gpu:
        print(i)
        sys.exit(0)
sys.exit(f"No EGL device found for GPU {gpu}")
PY
}


# =============================================================================
# Helpers
# =============================================================================

# Newest run dir <exp_name>_<date>_seed_<seed>, or empty.
latest_run_dir() {
    local name="$1" seed="$2"
    find "${RUNS_DIR}" -maxdepth 1 -type d -name "${name}_*_seed_${seed}" \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

# Highest checkpoint step in <run_dir>/weights_and_args, or empty.
latest_ckpt_step() {
    local run_dir="$1"
    [[ -n "${run_dir}" && -d "${run_dir}/weights_and_args" ]] || return 0
    find "${run_dir}/weights_and_args" -maxdepth 1 -name 'checkpoint_*.pth' \
        | sed -E 's/.*checkpoint_([0-9]+)\.pth/\1/' | sort -n | tail -n 1
}

# --total_timesteps the run was started with (from its args.json).
run_total_timesteps() {
    "${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_timesteps"])' \
        "$1/weights_and_args/args.json"
}

# True if an mpo_acme.py process with this exp_name and seed is alive
# (including ones started outside this script, e.g. an earlier launch).
is_running() {
    local name="$1" seed="$2"
    pgrep -f -- "${SCRIPT} .*--exp_name ${name} .*--seed ${seed}( |$)" > /dev/null
}

# Run mpo_acme.py once a GPU slot is free. Slots are flock'd files, so they are
# shared by every job (and every concurrent launch of this script) and released
# automatically when the job's subshell exits.
#   run_on_gpu <log_file> <mpo_acme.py args...>
run_on_gpu() {
    local log_file="$1"; shift
    (
        local gpu="" slot fd
        mkdir -p "${RUNS_DIR}/.gpu_slots"
        while [[ -z "${gpu}" ]]; do
            # Slot-major order: fill slot 0 on every GPU before slot 1, so
            # runs spread across GPUs instead of piling onto the first one.
            for ((slot=0; slot<PER_GPU; slot++)); do
                for g in "${GPU_LIST[@]}"; do
                    exec {fd}> "${RUNS_DIR}/.gpu_slots/gpu${g}_slot${slot}.lock"
                    if flock -n "${fd}"; then gpu="${g}"; break 2; fi
                    exec {fd}>&-
                done
            done
            [[ -n "${gpu}" ]] || sleep 30
        done

        local egl_device
        egl_device="$(egl_device_for_gpu "${gpu}")"
        echo "    -> GPU ${gpu} (EGL device ${egl_device}), log: ${log_file}"
        CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${egl_device}" \
            "${PY}" "${SCRIPT}" "$@" >> "${log_file}" 2>&1
    )
}


# =============================================================================
# Sim1 (one per seed). Sets SIM1_DIR and SIM1_STEP on success.
# =============================================================================

ensure_sim1() {
    local seed="$1"
    local target="${SIM1_TOTAL_TIMESTEPS//_/}"
    local log_file="${RUNS_DIR}/logs/${SIM1_EXP_NAME}_seed${seed}.log"
    local attempts=0

    if (( ! RESUME )); then
        echo "[seed ${seed}] Sim1: starting new run"
        run_on_gpu "${log_file}" \
            --exp_name "${SIM1_EXP_NAME}" --seed "${seed}" \
            --total_timesteps "${SIM1_TOTAL_TIMESTEPS}" \
            --runs_directory "${RUNS_DIR}" \
            --model_path "${SIM1_MODEL_PATH}" \
            "${COMMON_ARGS[@]}" || true
        local dir step
        dir="$(latest_run_dir "${SIM1_EXP_NAME}" "${seed}")"
        step="$(latest_ckpt_step "${dir}")"
        if [[ -z "${step}" ]] || (( step < target )); then
            echo "[seed ${seed}] ERROR: Sim1 stopped at step ${step:-0}/${target}; see ${log_file} (re-launch with --resume to continue it)" >&2
            return 1
        fi
        SIM1_DIR="${dir}"; SIM1_STEP="${step}"
        echo "[seed ${seed}] Sim1 done: ${dir} (step ${step})"
        return 0
    fi

    while true; do
        local dir step
        dir="$(latest_run_dir "${SIM1_EXP_NAME}" "${seed}")"
        step="$(latest_ckpt_step "${dir}")"

        if [[ -n "${step}" ]] && (( step >= target )); then
            SIM1_DIR="${dir}"; SIM1_STEP="${step}"
            echo "[seed ${seed}] Sim1 done: ${dir} (step ${step})"
            return 0
        fi

        if is_running "${SIM1_EXP_NAME}" "${seed}"; then
            echo "[seed ${seed}] Sim1 still training (${dir:-no dir yet}, step ${step:-0}/${target}); re-checking in ${POLL_SECONDS}s"
            sleep "${POLL_SECONDS}"
            continue
        fi

        # Not finished and not running: (re)start it, but don't loop forever on a crash.
        if (( attempts >= 2 )); then
            echo "[seed ${seed}] ERROR: Sim1 did not reach step ${target} after ${attempts} attempts; see ${log_file}" >&2
            return 1
        fi
        attempts=$((attempts + 1))

        if [[ -n "${step}" ]]; then
            echo "[seed ${seed}] Sim1 stopped at step ${step}; resuming ${dir}"
            run_on_gpu "${log_file}" \
                --exp_name "${SIM1_EXP_NAME}" --seed "${seed}" \
                --total_timesteps "${SIM1_TOTAL_TIMESTEPS}" \
                --runs_directory "${RUNS_DIR}" \
                --model_path "${SIM1_MODEL_PATH}" \
                --resume_in_place --weights_path "${dir}/weights_and_args" \
                "${COMMON_ARGS[@]}" || true
        else
            echo "[seed ${seed}] Sim1: starting from scratch"
            run_on_gpu "${log_file}" \
                --exp_name "${SIM1_EXP_NAME}" --seed "${seed}" \
                --total_timesteps "${SIM1_TOTAL_TIMESTEPS}" \
                --runs_directory "${RUNS_DIR}" \
                --model_path "${SIM1_MODEL_PATH}" \
                "${COMMON_ARGS[@]}" || true
        fi
    done
}


# =============================================================================
# Sim2 (one per variant x seed)
# =============================================================================

run_sim2() {
    local variant="$1" seed="$2"
    local name="${SIM2_PREFIX}_${variant}"
    local xml="${GEN_DIR}/${variant}.xml"
    local log_file="${RUNS_DIR}/logs/${name}_seed${seed}.log"
    [[ -f "${xml}" ]] || { echo "[seed ${seed}] ERROR: missing ${xml}" >&2; return 1; }

    # Without --resume, always start a new run (step stays empty).
    local dir="" step=""
    if (( RESUME )); then
        if is_running "${name}" "${seed}"; then
            echo "[seed ${seed}] ${variant}: already training, skipping"
            return 0
        fi
        dir="$(latest_run_dir "${name}" "${seed}")"
        step="$(latest_ckpt_step "${dir}")"
    fi

    if [[ -n "${step}" ]]; then
        # Resume an interrupted Sim2 run, keeping the total it was started with.
        local total
        total="$(run_total_timesteps "${dir}")"
        if (( step >= total )); then
            echo "[seed ${seed}] ${variant}: already finished (${dir}), skipping"
            return 0
        fi
        echo "[seed ${seed}] ${variant}: resuming ${dir} from step ${step}/${total}"
        run_on_gpu "${log_file}" \
            --exp_name "${name}" --seed "${seed}" \
            --total_timesteps "${total}" \
            --runs_directory "${RUNS_DIR}" \
            --model_path "${xml}" \
            --resume_in_place --weights_path "${dir}/weights_and_args" \
            "${COMMON_ARGS[@]}"
    else
        # Fresh Sim2: load Sim1's checkpoint AND its replay buffer
        # (mpo_acme.py loads replay_buffer.npz from --weights_path if present).
        local total=$(( SIM1_STEP + ${SIM2_STEPS//_/} ))
        echo "[seed ${seed}] ${variant}: warm start (with Sim1 buffer) from ${SIM1_DIR} step ${SIM1_STEP}, training to ${total}"
        run_on_gpu "${log_file}" \
            --exp_name "${name}" --seed "${seed}" \
            --total_timesteps "${total}" \
            --runs_directory "${RUNS_DIR}" \
            --model_path "${xml}" \
            --weights_path "${SIM1_DIR}/weights_and_args" \
            --checkpoint_step "${SIM1_STEP}" \
            "${COMMON_ARGS[@]}"
    fi
    echo "[seed ${seed}] ${variant}: done"
}


# Sim2-only mode: take the newest Sim1 run's latest checkpoint as it is now,
# without resuming, waiting for, or checking that Sim1 finished. Checkpoints
# written in the last minute are skipped in case one is still being saved.
# Sets SIM1_DIR and SIM1_STEP on success.
use_latest_sim1() {
    local seed="$1" dir step
    dir="$(latest_run_dir "${SIM1_EXP_NAME}" "${seed}")"
    if [[ -n "${dir}" && -d "${dir}/weights_and_args" ]]; then
        step="$(find "${dir}/weights_and_args" -maxdepth 1 -name 'checkpoint_*.pth' -mmin +1 \
            | sed -E 's/.*checkpoint_([0-9]+)\.pth/\1/' | sort -n | tail -n 1)"
    fi
    if [[ -z "${step:-}" ]]; then
        echo "[seed ${seed}] ERROR: no Sim1 checkpoint found for ${SIM1_EXP_NAME} seed ${seed} in ${RUNS_DIR}" >&2
        return 1
    fi
    SIM1_DIR="${dir}"; SIM1_STEP="${step}"
    local note=""
    is_running "${SIM1_EXP_NAME}" "${seed}" && note=" (Sim1 still training; using its current checkpoint)"
    echo "[seed ${seed}] Sim1: using ${dir} checkpoint ${step}${note}"
}


seed_pipeline() {
    local seed="$1"
    if [[ "${RUN_MODE}" == "sim_continual_learning" ]]; then
        use_latest_sim1 "${seed}" || return 1
    else
        ensure_sim1 "${seed}" || return 1
    fi
    [[ "${RUN_MODE}" == "sim" ]] && return 0

    local pids=() failed=0
    for variant in "${VARIANTS[@]}"; do
        run_sim2 "${variant}" "${seed}" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
    return "${failed}"
}


# =============================================================================
# Main
# =============================================================================

mkdir -p "${RUNS_DIR}/logs"

"${PY}" "${GEN_SCRIPT}" > /dev/null
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    mapfile -t VARIANTS < "${GEN_DIR}/manifest.txt"
fi

echo "Mode:     ${RUN_MODE}$( (( RESUME )) && echo ' (--resume)' || echo ' (new runs)')"
echo "Seeds:    ${SEEDS[*]}"
echo "Variants: ${VARIANTS[*]}"
echo "GPUs:     ${GPU_LIST[*]} (max ${PER_GPU} runs each)"

pids=()
for seed in "${SEEDS[@]}"; do
    seed_pipeline "${seed}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done

if [[ "${failed}" -ne 0 ]]; then
    echo "One or more runs failed; see ${RUNS_DIR}/logs/" >&2
    exit 1
fi

echo
echo "============================================================"
echo "All Mini Pi runs finished (mode: ${RUN_MODE})."
echo "Logs: ${RUNS_DIR}/logs/"
echo "============================================================"
