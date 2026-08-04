"""
python3 -m tuning.runner --entry agents.mpo.tune_mpo --name mpo_search --storage-dir runs/tuning --workers N --n-trials M
"""

from tuning.adapters import MPO
from tuning.cfg import TuningConfig
from tuning.common import SHARED_FIXED_CONFIG, SHARED_SEARCH_SPACE
from tuning.search_space import Cat, Float, Int

MPO_SEARCH_SPACE = {
    **SHARED_SEARCH_SPACE,
    "critic_type": Cat(["scalar", "categorical"]),
    "ensemble": Int(1, 4),
    "num_atoms": Cat([51, 101, 201]),
    "dual_lr": Float(1e-3, 1e-1, log=True),
    "epsilon_eta": Float(1e-2, 3.0, log=True),
    "epsilon_mu_kl": Float(1e-4, 1e-1, log=True),
    "epsilon_sigma_kl": Float(1e-7, 1e-4, log=True),
    "sample_action_num": Cat([32, 64, 128]),
    "td_horizon": Cat([1, 2, 3]),
    "samples_per_insert": Float(128.0, 4096.0, log=True),
    "target_policy_update_period": Cat([25, 50, 100, 200]),
    "target_critic_update_period": Cat([25, 50, 100, 200]),
    "policy_width": Cat([64, 128, 256]),
    "policy_depth": Int(2, 4),
    "critic_width": Cat([256, 512, 1024]),
    "critic_depth": Int(2, 6),
    "max_grad_norm": Float(1.0, 1000.0, log=True),
}

MPO_FIXED_CONFIG = {
    **SHARED_FIXED_CONFIG,
    "log_every_n_steps": 1000,
    "vmin": -500,
    "vmax": 20,
}


def get_tuning_setup() -> TuningConfig:
    return TuningConfig(
        space=MPO_SEARCH_SPACE,
        fixed_config={
            **MPO_FIXED_CONFIG,
            "exp_name": "tune_mpo",
        },
        adapter=MPO,
    )
