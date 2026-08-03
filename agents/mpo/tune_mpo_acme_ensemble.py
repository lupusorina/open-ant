"""
python3 -m tuning.runner --entry agents.mpo.tune_mpo_acme_ensemble --name mpo_acme_ensemble_search --storage-dir runs/tuning --workers N --n-trials M
"""

from tuning.adapters import MPO_ACME_ENSEMBLE
from tuning.cfg import TuningConfig
from tuning.search_space import Int

from .tune_mpo_acme import MPO_FIXED_CONFIG, MPO_SEARCH_SPACE

EMPO_SEARCH_SPACE = {
    **MPO_SEARCH_SPACE,
    "ensemble": Int(2, 4),
}

EMPO_FIXED_CONFIG = {
    **MPO_FIXED_CONFIG,
}


def get_tuning_setup() -> TuningConfig:
    return TuningConfig(
        space=EMPO_SEARCH_SPACE,
        fixed_config={
            **EMPO_FIXED_CONFIG,
            "exp_name": "tune_mpo_acme_ensemble",
        },
        adapter=MPO_ACME_ENSEMBLE,
    )
