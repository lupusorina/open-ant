"""
python3 -m tuning.runner --entry agents.dmpo.tune_dmpo_acme --name dmpo_acme_search --storage-dir runs/tuning --workers N --n-trials M
"""

from agents.mpo.tune_mpo_acme import MPO_FIXED_CONFIG, MPO_SEARCH_SPACE
from tuning.adapters import DMPO_ACME
from tuning.cfg import TuningConfig
from tuning.search_space import Cat

DMPO_SEARCH_SPACE = {
    **MPO_SEARCH_SPACE,
    "num_atoms": Cat([51, 101, 201]),
    "decouple_q_learning": Cat([True, False]),
}

DMPO_FIXED_CONFIG = {
    **MPO_FIXED_CONFIG,
    "policy_learning_starts": 2_000,
    "vmin": -500,
    "vmax": 20,
}


def get_tuning_setup() -> TuningConfig:
    return TuningConfig(
        space=DMPO_SEARCH_SPACE,
        fixed_config={
            **DMPO_FIXED_CONFIG,
            "exp_name": "tune_dmpo_acme",
        },
        adapter=DMPO_ACME,
    )
