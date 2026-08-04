"""
python3 -m tuning.runner --entry agents.sac.tune_sac --name sac_search --storage-dir runs/tuning --workers N --n-trials M
"""

from tuning.adapters import SAC
from tuning.cfg import TuningConfig
from tuning.common import SHARED_FIXED_CONFIG, SHARED_SEARCH_SPACE
from tuning.search_space import Cat, Float

# SAC hardcodes its layer sizes, so there is nothing to search there.
SAC_SEARCH_SPACE = {
    **SHARED_SEARCH_SPACE,
    "alpha_lr": Float(1e-4, 1e-2, log=True),
    "tau": Float(1e-3, 5e-2, log=True),
    "policy_frequency": Cat([1, 2, 4]),
    "target_network_frequency": Cat([1, 2, 4]),
}

SAC_FIXED_CONFIG = {
    **SHARED_FIXED_CONFIG,
    "autotune": True,
}


def get_tuning_setup() -> TuningConfig:
    return TuningConfig(
        space=SAC_SEARCH_SPACE,
        fixed_config={
            **SAC_FIXED_CONFIG,
            "exp_name": "tune_sac",
        },
        adapter=SAC,
    )
