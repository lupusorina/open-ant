"""
python3 -m tuning.runner --entry agents.mpo.tune_mpo_acme --name mpo_acme_search --storage-dir runs/tuning --workers N --n-trials M
"""

from tuning.adapters import MPO_ACME
from tuning.cfg import TuningConfig
from tuning.search_space import Cat, Float, Int

MPO_SEARCH_SPACE = {
    "policy_lr": Float(1e-5, 3e-3, log=True),
    "q_lr": Float(1e-5, 3e-3, log=True),
    "dual_lr": Float(1e-4, 1e-1, log=True),
    "epsilon_eta": Float(1e-3, 1.0, log=True),
    "epsilon_mu_kl": Float(1e-4, 1e-1, log=True),
    "epsilon_sigma_kl": Float(1e-7, 1e-4, log=True),
    "gamma": Float(0.8, 0.99999),
    "batch_size": Cat([128, 256, 512]),
    "sample_action_num": Cat([16, 32, 64]),
    "td_horizon": Cat([1, 3, 5]),
    "samples_per_insert": Float(64.0, 4096.0, log=True),
    "target_policy_update_period": Cat([25, 50, 100, 200]),
    "target_critic_update_period": Cat([25, 50, 100, 200]),
    "policy_width": Cat([128, 256, 512]),
    "policy_depth": Int(2, 4),
    "critic_width": Cat([256, 512, 1024]),
    "critic_depth": Int(2, 4),
    "use_layer_norm": Cat([True, False]),
    "max_grad_norm": Float(0.5, 100.0, log=True),
}

MPO_FIXED_CONFIG = {
    "total_timesteps": 50_000,
    "learning_starts": 2_000,
    "runs_directory": "runs/tuning_runs",
    "save_every_n_steps": 50_000,
    "log_every_n_steps": 1000,
    "cuda": True,
    "base_seed": 1000,
}


def get_tuning_setup() -> TuningConfig:
    return TuningConfig(
        space=MPO_SEARCH_SPACE,
        fixed_config={
            **MPO_FIXED_CONFIG,
            "exp_name": "tune_mpo_acme",
        },
        adapter=MPO_ACME,
    )
