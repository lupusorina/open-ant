"""
python3 -m tuning.runner --entry agents.mpo.tune_edmpo_stage1 --name edmpo_stage1 --storage-dir runs/tuning --workers N --n-trials M
"""

from agents.mpo.stage1_common import (
    STAGE1_DISTRIBUTIONAL_SPACE,
    STAGE1_FIXED_CONFIG,
    STAGE1_SEARCH_SPACE,
    STAGE1_STUDY_ATTRS,
)
from tuning.adapters import MPO
from tuning.cfg import TuningConfig
from tuning.search_space import Int

EDMPO_STAGE1_SEARCH_SPACE = {
    **STAGE1_SEARCH_SPACE,
    **STAGE1_DISTRIBUTIONAL_SPACE,
    "ensemble": Int(2, 3),
}

EDMPO_STAGE1_FIXED_CONFIG = {
    **STAGE1_FIXED_CONFIG,
    "critic_type": "categorical",
    "exp_name": "tune_edmpo_stage1",
}


def get_tuning_setup() -> TuningConfig:
    return TuningConfig(
        space=EDMPO_STAGE1_SEARCH_SPACE,
        fixed_config=EDMPO_STAGE1_FIXED_CONFIG,
        adapter=MPO,
        study_attrs=STAGE1_STUDY_ATTRS,
    )
