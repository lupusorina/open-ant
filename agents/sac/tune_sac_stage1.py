"""
python3 -m tuning.runner --entry agents.sac.tune_sac_stage1 --name sac_stage1 --journal stage1 --storage-dir runs/tuning --workers N --n-trials M
"""

from tuning.adapters import SAC
from tuning.cfg import TuningConfig
from tuning.common import (
    STAGE1_DOMAIN_RANDOMIZATION,
    STAGE1_ENV_CONFIG,
    objective_revision,
    scoring_study_attrs,
    seed_scheme,
)
from tuning.search_space import Cat, Float, Int

SAC_STAGE1_SEARCH_SPACE = {
    "policy_lr": Float(2e-4, 1e-2, log=True),
    "q_lr": Float(3e-4, 2e-3, log=True),
    "gamma": Float(0.84, 0.96),
    "alpha_lr": Float(3e-4, 1e-2, log=True),
    "tau": Float(2e-3, 0.2, log=True),
    "batch_size": Int(96, 320, step=32),
    "policy_frequency": Cat([1, 2]),
    "target_network_frequency": Cat([1, 2, 4]),
    "policy_width": Int(48, 192, step=16),
    "policy_depth": Int(2, 4),
    "critic_width": Int(256, 1536, step=256),
    "critic_depth": Int(2, 5),
}

SAC_STAGE1_FIXED_CONFIG = {
    **STAGE1_ENV_CONFIG,
    "autotune": True,
    "use_layer_norm": True,
    "exp_name": "tune_sac_stage1",
}

SAC_SEEDS_PER_TRIAL = 4  # 3

SAC_STAGE1_STUDY_ATTRS = scoring_study_attrs(seeds_per_trial=SAC_SEEDS_PER_TRIAL)
SAC_STAGE1_STUDY_ATTRS.update({
    "domain_randomization": STAGE1_DOMAIN_RANDOMIZATION,
    "seed_scheme": seed_scheme(
        SAC_STAGE1_STUDY_ATTRS["paired_seeds"],
        SAC_STAGE1_STUDY_ATTRS["seeds_per_trial"],
    ),
    "space_revision": "sac-stage1-rev2",
    "objective_revision": objective_revision(SAC_STAGE1_STUDY_ATTRS),
})


def get_tuning_setup() -> TuningConfig:
    return TuningConfig(
        space=SAC_STAGE1_SEARCH_SPACE,
        fixed_config=SAC_STAGE1_FIXED_CONFIG,
        adapter=SAC,
        study_attrs=SAC_STAGE1_STUDY_ATTRS,
        seeds_per_trial=SAC_SEEDS_PER_TRIAL,
    )
