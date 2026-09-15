"""Search dimensions and settings that every agent in this repository shares."""

from dataclasses import MISSING, fields

from .cfg import TuningConfig
from .search_space import Cat, Float, Int

SHARED_SEARCH_SPACE = {
    "policy_lr": Float(0.0005, 0.05, log=True),
    "q_lr": Float(0.0005, 0.005, log=True),
    "gamma": Float(0.9, 0.999, log=True),
    "batch_size": Int(64, 320, step=32),
    "use_layer_norm": Cat([True, False]),
}

SHARED_FIXED_CONFIG = {
    "total_timesteps": 50_000,
    "save_every_n_steps": 50_000,
    "learning_starts": 2_500,
    "runs_directory": "runs/tuning_runs",
    "cuda": True,
    "base_seed": 1024,
}

STAGE1_ENV_CONFIG = {
    "env_id": "SimEmbodiedAntDR",
    "total_timesteps": 40_000,
    "learning_starts": 2_500,
    "save_every_n_steps": 40_000,
    "runs_directory": "runs/tuning_runs",
    "cuda": True,
    "base_seed": 1024,
    "dt": 0.12,
    "task_type": "back_and_forth",
    "radius_back_and_forth": 0.3,
    "origin_back_and_forth": [0.7, -0.25],
    "reward_scale": 100.0,
}

STAGE1_DOMAIN_RANDOMIZATION = {
    "mass": [0.7, 2.2],
    "friction": [0.45, 0.9],
    "timeconst": [0.08, 0.45],
}

SCORING_ATTR_KEYS = (
    "seeds_per_trial",
    "paired_seeds",
    "check_percentile",
    "pruner_percentile",
    "pruner_warmup_fraction",
    "last_fraction",
    "eval_weight",
    "tail_weight",
    "eval_env_id",
    "eval_episodes",
    "eval_seed_start",
    "eval_max_steps",
    "duration_lambda",
    "duration_ref_hours",
    "stability_k",
    "seed_cap_hours",
    "cap_projection_hours",
    "cap_projection_min_hours",
)

_CFG_DEFAULTS = {
    f.name: f.default for f in fields(TuningConfig) if f.default is not MISSING
}


def scoring_study_attrs(**overrides) -> dict:
    attrs = {key: _CFG_DEFAULTS[key] for key in SCORING_ATTR_KEYS}
    for key, value in overrides.items():
        if key not in attrs:
            raise KeyError(
                f"{key!r} is not a scoring attr; add it to SCORING_ATTR_KEYS"
            )
        attrs[key] = value
    return attrs


def seed_scheme(paired_seeds: bool, seeds_per_trial: int) -> str:
    if paired_seeds:
        return "base_seed + seed_index"
    return f"base_seed + {seeds_per_trial}*trial + seed_index"


def seed_for(
    base_seed: int,
    trial_number: int,
    seed_index: int,
    *,
    paired_seeds: bool,
    seeds_per_trial: int,
) -> int:
    if paired_seeds:
        return base_seed + seed_index
    return base_seed + seeds_per_trial * trial_number + seed_index


def objective_revision(attrs: dict, *, phase: str = "stage1") -> str:
    per_hour = attrs["duration_lambda"] / attrs["duration_ref_hours"]
    tail = f"train_tail(last {attrs['last_fraction']:g})"
    if phase == "continual":
        deploy_weight = attrs["deploy_weight"]
        head = (
            f"J_seed = {deploy_weight:g}*deploy(first {attrs['deploy_window_steps']}"
            f" of {attrs['continual_steps']} steps)"
            f" + {1.0 - deploy_weight:g}*{tail}"
        )
    else:
        head = (
            f"J_seed = {attrs['eval_weight']:g}*eval({attrs['eval_env_id']},"
            f" {attrs['eval_episodes']}x{attrs['eval_max_steps']} from seed"
            f" {attrs['eval_seed_start']})"
            f" + {attrs['tail_weight']:g}*{tail}"
        )
    return (
        f"{head} - {per_hour:g}*hours; "
        f"J_trial = mean(J_seeds) - {attrs['stability_k']:g}*std(J_seeds) "
        f"over {attrs['seeds_per_trial']} seeds "
        f"({seed_scheme(attrs['paired_seeds'], attrs['seeds_per_trial'])}); "
        "train tail scaled by reward_scale to match eval units"
    )
