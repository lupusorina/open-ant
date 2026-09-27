"""
python3 -m tuning.runner --entry agents.mpo.tune_mpo_stage1 --name mpo_stage1 --storage-dir runs/tuning --workers N --n-trials M
"""

from agents.mpo.stage1_common import STAGE1_FIXED_CONFIG, STAGE1_SEARCH_SPACE, STAGE1_STUDY_ATTRS
from tuning.adapters import MPO
from tuning.cfg import TuningConfig

MPO_STAGE1_SEARCH_SPACE = dict(STAGE1_SEARCH_SPACE)

MPO_STAGE1_FIXED_CONFIG = {
    **STAGE1_FIXED_CONFIG,
    "critic_type": "scalar",
    "ensemble": 1,
    "exp_name": "tune_mpo_stage1",
}


def get_tuning_setup() -> TuningConfig:
    return TuningConfig(
        space=MPO_STAGE1_SEARCH_SPACE,
        fixed_config=MPO_STAGE1_FIXED_CONFIG,
        adapter=MPO,
        study_attrs=STAGE1_STUDY_ATTRS,
    )
