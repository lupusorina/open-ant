"""
python3 -m tuning.runner --entry agents.mpo.tune_mpo --name mpo_search --storage-dir runs/tuning --workers N --n-trials M
"""

from tuning.adapters import MPO
from tuning.cfg import TuningConfig
from tuning.common import SHARED_FIXED_CONFIG, SHARED_SEARCH_SPACE
from tuning.search_space import Float, Int

MPO_SEARCH_SPACE = {
    **SHARED_SEARCH_SPACE,
    "ensemble": Int(2, 5),
    "epsilon_mu_kl": Float(0.005, 0.5, log=True),
    "epsilon_eta": Float(0.1, 5.0, log=True),
    "max_grad_norm": Float(0.1, 100.0, log=True),
    "samples_per_insert": Float(512.0, 4096.0, log=True),
    "critic_depth": Int(4, 6),
    "sample_action_num": Int(40, 120, step=8),
    "critic_width": Int(512, 2048, step=256),
    "policy_width": Int(16, 128, step=16),
    "policy_depth": Int(3, 7),
    "num_atoms": Int(81, 401, step=40),
    "dual_lr": Float(0.002, 0.2, log=True),
    "epsilon_sigma_kl": Float(0.00000001, 0.001, log=True),
    "target_policy_update_period": Int(40, 400, step=20),
    "target_critic_update_period": Int(40, 400, step=20),
    "vmin": Float(-5000.0, -5.0),
    "vmax": Float(1, 1000),
}

MPO_FIXED_CONFIG = {
    **SHARED_FIXED_CONFIG,
    "log_every_n_steps": 1000,
    "critic_type": "scalar",
    "td_horizon": 1,
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
