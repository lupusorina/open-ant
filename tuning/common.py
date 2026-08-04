"""Search dimensions and settings that every agent in this repository shares."""

from .search_space import Cat, Float

SHARED_SEARCH_SPACE = {
    "policy_lr": Float(1e-4, 1e-2, log=True),
    "q_lr": Float(1e-4, 1e-2, log=True),
    "gamma": Float(0.85, 0.999),
    "batch_size": Cat([256, 512, 1024]),
}

SHARED_FIXED_CONFIG = {
    "total_timesteps": 40_000,
    "learning_starts": 2_000,
    "use_layer_norm": True,
    "runs_directory": "runs/tuning_runs",
    "save_every_n_steps": 50_000,
    "cuda": True,
    "base_seed": 1000,
}
