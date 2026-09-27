"""
python3 -m tuning.runner --entry agents.mpo.tune_empo_stage1 --name empo_stage1 --storage-dir runs/tuning --workers N --n-trials M
"""

from agents.mpo.stage1_common import STAGE1_FIXED_CONFIG, STAGE1_SEARCH_SPACE, STAGE1_STUDY_ATTRS
from tuning.adapters import MPO
from tuning.cfg import TuningConfig
from tuning.search_space import Int

EMPO_STAGE1_SEARCH_SPACE = {
    **STAGE1_SEARCH_SPACE,
    "ensemble": Int(2, 3),
}

EMPO_STAGE1_FIXED_CONFIG = {
    **STAGE1_FIXED_CONFIG,
    "critic_type": "scalar",
    "exp_name": "tune_empo_stage1",
}


def get_tuning_setup() -> TuningConfig:
    return TuningConfig(
        space=EMPO_STAGE1_SEARCH_SPACE,
        fixed_config=EMPO_STAGE1_FIXED_CONFIG,
        adapter=MPO,
        study_attrs=STAGE1_STUDY_ATTRS,
    )
