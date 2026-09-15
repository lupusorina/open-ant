"""
python3 -m tuning.runner --entry agents.mpo.tune_dmpo_stage1 --name dmpo_stage1 --storage-dir runs/tuning --workers N --n-trials M
"""

from agents.mpo.stage1_common import (
    STAGE1_DISTRIBUTIONAL_SPACE,
    STAGE1_FIXED_CONFIG,
    STAGE1_SEARCH_SPACE,
    STAGE1_STUDY_ATTRS,
)
from tuning.adapters import MPO
from tuning.cfg import TuningConfig

DMPO_STAGE1_SEARCH_SPACE = {
    **STAGE1_SEARCH_SPACE,
    **STAGE1_DISTRIBUTIONAL_SPACE,
}

DMPO_STAGE1_FIXED_CONFIG = {
    **STAGE1_FIXED_CONFIG,
    "critic_type": "categorical",
    "ensemble": 1,
    "exp_name": "tune_dmpo_stage1",
}


def get_tuning_setup() -> TuningConfig:
    return TuningConfig(
        space=DMPO_STAGE1_SEARCH_SPACE,
        fixed_config=DMPO_STAGE1_FIXED_CONFIG,
        adapter=MPO,
        study_attrs=STAGE1_STUDY_ATTRS,
    )
