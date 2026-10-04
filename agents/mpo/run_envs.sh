#!/usr/bin/env bash
# SAC or MPO (any critic) on swimmer / reacher / pusher / mini_pi, Sim1 -> Sim2 sweep.
# Built from run_minipiall_threshold.sh (MPO) and run_minipi_sac_threshold.sh (SAC).
#   Sim1: ENV's base XML, one run per seed, until the eval return reaches REWARD_THRESHOLD
#         or SIM1_TOTAL_TIMESTEPS.
#   Sim2: per (variant, seed), continue from Sim1's best eval on a heavier, lower-friction model:
#         swimmer/reacher/pusher: <env>.jinja.xml rendered at each SIM2_SHIFTS entry;
#         mini_pi: the XMLs in sim/assets/hightorque/generated/ (manifest.txt).
#
# Default: every launch starts NEW timestamped runs. --resume picks up existing runs instead:
#   Sim1 finished -> Sim2 | Sim1 still running -> wait | Sim1 stopped part-way -> resume it
#   Sim2 finished/running -> skip | Sim2 stopped part-way -> resume it
# Seeds run in parallel; each seed's Sim2 variants start in parallel once its Sim1 is done,
# capped at PER_GPU runs per GPU.
#
# Usage:
#   bash run_envs.sh                          # Sim1, then Sim2 (default)
#   bash run_envs.sh sim                      # Sim1 only
#   bash run_envs.sh sim_continual_learning   # Sim2 only, from the newest Sim1 checkpoint as-is
#   bash run_envs.sh [mode] --resume          # resume existing runs instead of starting new ones
# Logs: ${RUNS_DIR}/logs/<exp_name>_seed<seed>.log

set -euo pipefail
cd "$(dirname "$0")"

# ======================== edit these ========================
ENV="pusher"                # swimmer | reacher | pusher | mini_pi
ALGO="qmpo"                   # sac | mpo (scalar critic) | dmpo (categorical) | qmpo (quantile)
ENSEMBLE=1                   # MPO variants only: number of critics
GPU_LIST=(2)                 # physical GPU ids (nvidia-smi numbering)
PER_GPU=4                    # max concurrent runs this script starts per GPU
NUM_SEEDS=1                  # runs seeds 1..NUM_SEEDS
# Sim1 stops once the best deterministic eval return (mean of N_EVAL_EPISODES fixed-seed
# episodes every save_every_n_steps; ../sb3_eval.py, ../early_stopping.py) reaches this.
# Reaching it writes <run_dir>/converged.json (= "Sim1 finished"). Empty = no eval, no early
# stop: Sim1 always runs SIM1_TOTAL_TIMESTEPS. Returns differ a lot per env (Reacher/Pusher
# returns are negative), so set it for the env you pick.
REWARD_THRESHOLD=-10
N_EVAL_EPISODES=5
RUNS_DIR="/data2/serenaliu_data/2_${ALGO}_${ENV}"
SIM1_TOTAL_TIMESTEPS=1000000
SIM2_STEPS=1500000           # Sim2 runs (start step) -> (start step + SIM2_STEPS)
# 1 = Sim2 starts from Sim1's best eval (weights + buffer + step as they were then).
# No best checkpoint: last checkpoint (with a warning).
SIM2_LOAD_BEST=1
# swimmer / reacher / pusher: one Sim2 variant per shift s, rendered from <env>.jinja.xml
# (mass x(1+s); friction x(1-s), swimmer: viscosity x(1-s)) into ${RUNS_DIR}/xml/.
SIM2_SHIFTS=(0.2 0.4 0.6 0.8)
# mini_pi: Sim2 variants from its manifest; empty = all, e.g. (mini_pi_mass_fric_20 mini_pi_mass_fric_80)
VARIANTS=()
# ============================================================

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
        echo "Usage: bash run_envs.sh [sim|sim_continual_learning|sim_then_continual] [--resume]" >&2
        exit 1
        ;;
esac

# ---- Per-env settings ----
# Sim2 variants come from one of:
#   SIM2_TEMPLATE: jinja template rendered at launch, one XML per SIM2_SHIFTS entry
#   GEN_DIR:       folder with manifest.txt + <variant>.xml (GEN_SCRIPT creates it if missing)
SIM2_TEMPLATE="" GEN_DIR="" GEN_SCRIPT=""
case "${ENV}" in
    swimmer)
        ENV_ID="Swimmer-v5"
        SIM1_MODEL_PATH="../../sim/assets/swimmer/swimmer.xml"
        SIM2_TEMPLATE="../../sim/assets/swimmer/swimmer.jinja.xml"
        ;;
    reacher)
        ENV_ID="Reacher-v5"
        SIM1_MODEL_PATH="../../sim/assets/reacher/reacher.xml"
        SIM2_TEMPLATE="../../sim/assets/reacher/reacher.jinja.xml"
        ;;
    pusher)
        ENV_ID="Pusher-v5"
        SIM1_MODEL_PATH="../../sim/assets/pusher/pusher.xml"
        SIM2_TEMPLATE="../../sim/assets/pusher/pusher.jinja.xml"
        ;;
    mini_pi)
        ENV_ID="MiniPiWalk-v0"
        SIM1_MODEL_PATH="../../sim/assets/hightorque/scene.xml"
        GEN_DIR="../../sim/assets/hightorque/generated"
        GEN_SCRIPT="../../sim/assets/hightorque/make_mini_pi_variants.py"
        ;;
    *)
        echo "ERROR: unknown ENV '${ENV}' (swimmer | reacher | pusher | mini_pi)" >&2
        exit 1
        ;;
esac

SIM1_EXP_NAME="${ALGO}_${ENV}"
SIM2_PREFIX="continual_${ALGO}_${ENV}"   # Sim2 exp_name = ${SIM2_PREFIX}_<variant>

SIM1_EARLY_STOP_ARGS=()
[[ -n "${REWARD_THRESHOLD}" ]] && \
    SIM1_EARLY_STOP_ARGS=(--stop_reward_threshold "${REWARD_THRESHOLD}" --n_eval_episodes "${N_EVAL_EPISODES}")

SEEDS=($(seq 1 "${NUM_SEEDS}"))

# ---- Per-algorithm settings ----
case "${ALGO}" in
    sac)
        SCRIPT="sac_cleanrl_threshold.py"
        COMMON_ARGS=(
            --env_id "${ENV_ID}"
            --render_mode rgb_array
            --cuda
            --gamma 0.99                 # per step, same as MPO
            --num_envs 1
            --save_every_n_steps 25000
            --capture_video
            --capture_video_steps 1000
            --capture_video_every 50000
            --learning_starts 5000
        )
        ;;
    mpo|dmpo|qmpo)
        SCRIPT="mpo_acme_threshold.py"
        case "${ALGO}" in
            mpo)  CRITIC_TYPE="scalar" ;;
            dmpo) CRITIC_TYPE="categorical" ;;
            qmpo) CRITIC_TYPE="quantile" ;;
        esac
        COMMON_ARGS=(
            --env_id "${ENV_ID}"
            --render_mode rgb_array
            --cuda
            --critic_type "${CRITIC_TYPE}"
            --ensemble "${ENSEMBLE}"
            --gamma 0.99
            --dual_lr 1e-2
            --policy_lr 3e-4
            --q_lr 3e-4
            --batch_size 256
            --td_horizon 4
            --critic_layer_sizes 256 256 256
            --policy_init_scale 0.5
            --learning_starts 1000
            --epsilon_mu_kl 0.01
            --samples_per_insert 64 # 256
            --sample_action_num 20
            --policy_min_scale 1e-6
            --save_every_n_steps 25000
            --log_every_n_steps 10000
            --capture_video
            --capture_video_steps 1000
            --capture_video_every 50000
        )
        ;;
    *)
        echo "ERROR: unknown ALGO '${ALGO}' (sac | mpo | dmpo | qmpo)" >&2
        exit 1
        ;;
esac

POLL_SECONDS=600             # how often to re-check a Sim1 that is still training

# Headless MuJoCo rendering via EGL.
export MUJOCO_GL=egl
PY="${PY:-python3}"

# Make CUDA ordinals match nvidia-smi indices.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# EGL device order != nvidia-smi order here: find the EGL index of the requested GPU.
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


# ---- Helpers ----
# Checkpoint layout differs per script:
#   MPO: <run_dir>/weights_and_args/{checkpoint_<step>.pth, args.json},
#        <run_dir>/{best_checkpoint.pth, best_replay_buffer.npz}
#   SAC: <run_dir>/{weights.pth, args.json, best_weights.pth, best_replay_buffer/}

# Newest run dir <exp_name>_<date>_seed_<seed>, or empty.
latest_run_dir() {
    local name="$1" seed="$2"
    find "${RUNS_DIR}" -maxdepth 1 -type d -name "${name}_*_seed_${seed}" \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

# global_step stored in a .pth checkpoint.
pth_step() {
    "${PY}" -c 'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["global_step"])' "$1"
}

# What --weights_path points at for a run dir.
weights_dir() {
    if [[ "${ALGO}" == "sac" ]]; then echo "$1"; else echo "$1/weights_and_args"; fi
}

# Latest saved step of a run (MPO: highest checkpoint_<step>.pth; SAC: weights.pth), or empty.
# Extra find args (e.g. -mmin +1) skip MPO checkpoints still being written.
ckpt_step() {
    local run_dir="$1"; shift
    [[ -n "${run_dir}" ]] || return 0
    if [[ "${ALGO}" == "sac" ]]; then
        [[ -f "${run_dir}/weights.pth" ]] || return 0
        pth_step "${run_dir}/weights.pth"
    else
        [[ -d "${run_dir}/weights_and_args" ]] || return 0
        find "${run_dir}/weights_and_args" -maxdepth 1 -name 'checkpoint_*.pth' "$@" \
            | sed -E 's/.*checkpoint_([0-9]+)\.pth/\1/' | sort -n | tail -n 1
    fi
}

# --total_timesteps the run was started with (from its args.json).
run_total_timesteps() {
    "${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_timesteps"])' \
        "$(weights_dir "$1")/args.json"
}

# True if a run with this exp_name and seed is alive (from any launch).
is_running() {
    local name="$1" seed="$2"
    pgrep -f -- "${SCRIPT} .*--exp_name ${name} .*--seed ${seed}( |$)" > /dev/null
}

# run_on_gpu <log_file> <args...>: run once a GPU slot (flock'd file, shared across launches) is free.
run_on_gpu() {
    local log_file="$1"; shift
    (
        local gpu="" slot fd
        mkdir -p "${RUNS_DIR}/.gpu_slots"
        while [[ -z "${gpu}" ]]; do
            # Fill slot 0 on every GPU before slot 1.
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


# ---- Sim1 (one per seed). Sets SIM1_DIR and SIM1_STEP on success. ----

ensure_sim1() {
    local seed="$1"
    local target="${SIM1_TOTAL_TIMESTEPS//_/}"
    local log_file="${RUNS_DIR}/logs/${SIM1_EXP_NAME}_seed${seed}.log"
    local sim1_args=(
        --exp_name "${SIM1_EXP_NAME}" --seed "${seed}"
        --total_timesteps "${SIM1_TOTAL_TIMESTEPS}"
        --runs_directory "${RUNS_DIR}"
        --model_path "${SIM1_MODEL_PATH}"
        "${SIM1_EARLY_STOP_ARGS[@]}"
        "${COMMON_ARGS[@]}"
    )
    local attempts=0 dir step

    if (( ! RESUME )); then
        echo "[seed ${seed}] Sim1: starting new run"
        run_on_gpu "${log_file}" "${sim1_args[@]}" || true
        dir="$(latest_run_dir "${SIM1_EXP_NAME}" "${seed}")"
        step="$(ckpt_step "${dir}")"
        if [[ -z "${step}" ]] || { (( step < target )) && [[ ! -f "${dir}/converged.json" ]]; }; then
            echo "[seed ${seed}] ERROR: Sim1 stopped at step ${step:-0}/${target}; see ${log_file} (re-launch with --resume to continue it)" >&2
            return 1
        fi
        SIM1_DIR="${dir}"; SIM1_STEP="${step}"
        echo "[seed ${seed}] Sim1 done: ${dir} (step ${step})"
        return 0
    fi

    while true; do
        dir="$(latest_run_dir "${SIM1_EXP_NAME}" "${seed}")"
        step="$(ckpt_step "${dir}")"

        if [[ -n "${step}" ]] && { (( step >= target )) || [[ -f "${dir}/converged.json" ]]; }; then
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
            run_on_gpu "${log_file}" "${sim1_args[@]}" \
                --resume_in_place --weights_path "$(weights_dir "${dir}")" || true
        else
            echo "[seed ${seed}] Sim1: starting from scratch"
            run_on_gpu "${log_file}" "${sim1_args[@]}" || true
        fi
    done
}

# Sim2-only mode: newest Sim1 checkpoint as it is now (skips MPO ones <1 min old). Sets SIM1_DIR/SIM1_STEP.
use_latest_sim1() {
    local seed="$1" dir step
    dir="$(latest_run_dir "${SIM1_EXP_NAME}" "${seed}")"
    step="$(ckpt_step "${dir}" -mmin +1)"
    if [[ -z "${step}" ]]; then
        echo "[seed ${seed}] ERROR: no Sim1 checkpoint found for ${SIM1_EXP_NAME} seed ${seed} in ${RUNS_DIR}" >&2
        return 1
    fi
    SIM1_DIR="${dir}"; SIM1_STEP="${step}"
    local note=""
    is_running "${SIM1_EXP_NAME}" "${seed}" && note=" (Sim1 still training; using its current checkpoint)"
    echo "[seed ${seed}] Sim1: using ${dir} checkpoint ${step}${note}"
}


# ---- Sim2 (one per variant x seed) ----

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
        step="$(ckpt_step "${dir}")"
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
            --resume_in_place --weights_path "$(weights_dir "${dir}")" \
            "${COMMON_ARGS[@]}"
        echo "[seed ${seed}] ${variant}: done"
        return
    fi

    # Fresh Sim2: Sim1's best-eval checkpoint + buffer + step (or the last ones).
    local best_ckpt best_buffer_ok=0 load_args=()
    local start="${SIM1_STEP}" source="last checkpoint + last buffer"
    if [[ "${ALGO}" == "sac" ]]; then
        best_ckpt="${SIM1_DIR}/best_weights.pth"
        [[ -d "${SIM1_DIR}/best_replay_buffer" ]] && best_buffer_ok=1
    else
        best_ckpt="${SIM1_DIR}/best_checkpoint.pth"
        [[ -f "${SIM1_DIR}/best_replay_buffer.npz" ]] && best_buffer_ok=1
        load_args=(--checkpoint_step "${SIM1_STEP}")
    fi
    if (( SIM2_LOAD_BEST )); then
        if [[ -f "${best_ckpt}" ]]; then
            load_args=(--load_best) source="best checkpoint + last buffer"
            if (( best_buffer_ok )); then
                start="$(pth_step "${best_ckpt}")" source="best checkpoint + best buffer"
            fi
        else
            echo "[seed ${seed}] WARNING: ${best_ckpt} not found; ${variant} starts from the last checkpoint" >&2
        fi
    fi
    local total=$(( start + ${SIM2_STEPS//_/} ))
    echo "[seed ${seed}] ${variant}: warm start from ${SIM1_DIR} ${source}, steps ${start} -> ${total}"
    run_on_gpu "${log_file}" \
        --exp_name "${name}" --seed "${seed}" \
        --total_timesteps "${total}" \
        --runs_directory "${RUNS_DIR}" \
        --model_path "${xml}" \
        --weights_path "$(weights_dir "${SIM1_DIR}")" \
        "${load_args[@]}" \
        "${COMMON_ARGS[@]}"
    echo "[seed ${seed}] ${variant}: done"
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


# ---- Main ----

mkdir -p "${RUNS_DIR}/logs"

if [[ "${RUN_MODE}" != "sim" ]]; then
    if [[ -n "${SIM2_TEMPLATE}" ]]; then
        # Render one XML per shift into ${RUNS_DIR}/xml/ (kept next to the runs that use it).
        # Re-rendering on --resume rewrites identical files. Absolute: gymnasium resolves an
        # xml_file not starting with "/" or "." against its own assets folder.
        GEN_DIR="$(realpath -m "${RUNS_DIR}")/xml"
        mkdir -p "${GEN_DIR}"
        mapfile -t VARIANTS < <("${PY}" - "${SIM2_TEMPLATE}" "${GEN_DIR}" "${ENV}" "${SIM2_SHIFTS[@]}" <<'PY'
import os, sys
import jinja2, mujoco
template, out_dir, env, shifts = sys.argv[1], sys.argv[2], sys.argv[3], [float(s) for s in sys.argv[4:]]
with open(template) as f:
    tmpl = jinja2.Template(f.read())
for s in shifts:
    if not 0.0 <= s < 1.0:
        sys.exit(f"shift must be in [0, 1), got {s}")
    xml = tmpl.render(shift=s)
    mujoco.MjModel.from_xml_string(xml)  # fail here, not mid-training, if it doesn't compile
    name = f"{env}_shift_{round(s * 100):d}"
    with open(os.path.join(out_dir, f"{name}.xml"), "w") as f:
        f.write(xml)
    print(name)
PY
        )
        [[ ${#VARIANTS[@]} -eq ${#SIM2_SHIFTS[@]} ]] || {
            echo "ERROR: rendering ${SIM2_TEMPLATE} failed" >&2
            exit 1
        }
    else
        if [[ ! -f "${GEN_DIR}/manifest.txt" && -n "${GEN_SCRIPT}" ]]; then
            "${PY}" "${GEN_SCRIPT}" > /dev/null
        fi
        if [[ ${#VARIANTS[@]} -eq 0 ]]; then
            [[ -f "${GEN_DIR}/manifest.txt" ]] || {
                echo "ERROR: ${GEN_DIR}/manifest.txt not found" >&2
                exit 1
            }
            mapfile -t VARIANTS < "${GEN_DIR}/manifest.txt"
        fi
    fi
fi

echo "Env:      ${ENV} (${ENV_ID}, ${SIM1_MODEL_PATH})"
echo "Algo:     ${ALGO} (${SCRIPT})"
echo "Mode:     ${RUN_MODE}$( (( RESUME )) && echo ' (--resume)' || echo ' (new runs)')"
echo "Stop:     ${REWARD_THRESHOLD:+eval return >= ${REWARD_THRESHOLD} or }${SIM1_TOTAL_TIMESTEPS} steps"
echo "Seeds:    ${SEEDS[*]}"
echo "Variants: ${VARIANTS[*]:-(none)}"
echo "GPUs:     ${GPU_LIST[*]} (max ${PER_GPU} runs each)"
echo "Runs dir: ${RUNS_DIR}"

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
echo "All ${ALGO} ${ENV} runs finished (mode: ${RUN_MODE})."
echo "Logs: ${RUNS_DIR}/logs/"
echo "============================================================"
