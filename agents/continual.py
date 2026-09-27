"""
python3 -m tuning.runner --entry agents.continual --setup-arg best_performer.json --name empo_continual --journal stage3 --storage-dir runs/tuning --workers N --n-trials M
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import glob
import json
import os

from tuning.adapters import MPO, SAC, AgentAdapter
from tuning.cfg import TuningConfig
from tuning.common import objective_revision, scoring_study_attrs, seed_scheme
from tuning.search_space import Int, SpaceSpec

CONTINUAL_STEPS = 15_000

CONTINUAL_SEEDS_PER_TRIAL = 5

MPO_CONTINUAL_KEYS = (
    "policy_lr",
    "q_lr",
    "dual_lr",
    "epsilon_eta",
    "epsilon_mu_kl",
    "epsilon_sigma_kl",
    "target_policy_update_period",
    "target_critic_update_period",
    "samples_per_insert",
    "batch_size",
    "max_grad_norm",
)

SAC_CONTINUAL_KEYS = (
    "policy_lr",
    "q_lr",
    "alpha_lr",
    "tau",
    "batch_size",
    "policy_frequency",
    "target_network_frequency",
)


@dataclass(frozen=True)
class Family:
    adapter: AgentAdapter
    stage1_space: SpaceSpec
    stage1_fixed_config: Dict[str, Any]
    continual_keys: Tuple[str, ...]

    def space(self) -> SpaceSpec:
        space = {k: self.stage1_space[k] for k in self.continual_keys}
        space["learning_starts"] = Int(0, 2000)
        return space


def _families() -> Dict[str, Family]:
    from agents.mpo.stage1_common import STAGE1_FIXED_CONFIG, STAGE1_SEARCH_SPACE
    from agents.sac.tune_sac_stage1 import (
        SAC_STAGE1_FIXED_CONFIG,
        SAC_STAGE1_SEARCH_SPACE,
    )

    mpo = Family(MPO, STAGE1_SEARCH_SPACE, STAGE1_FIXED_CONFIG, MPO_CONTINUAL_KEYS)
    sac = Family(
        SAC, SAC_STAGE1_SEARCH_SPACE, SAC_STAGE1_FIXED_CONFIG, SAC_CONTINUAL_KEYS
    )
    return {"mpo": mpo, "empo": mpo, "dmpo": mpo, "edmpo": mpo, "sac": sac}


def family_for(algo: str) -> Family:
    families = _families()
    if algo not in families:
        raise ValueError(
            f"no continual family for algo {algo!r}; known: {sorted(families)}"
        )
    return families[algo]


def get_tuning_setup(setup_arg: str) -> TuningConfig:
    with open(setup_arg) as f:
        spec = json.load(f)

    for directory in spec["checkpoint_dirs"]:
        if not os.path.isdir(directory):
            raise ValueError(f"checkpoint dir does not exist: {directory}")
        if not glob.glob(os.path.join(directory, "checkpoint_*.pth")):
            raise ValueError(f"no checkpoint_*.pth in checkpoint dir: {directory}")

    algo = spec["algo"]
    family = family_for(algo)
    space = family.space()

    best_performer = {
        k: v for k, v in spec["best_performer_config"].items() if k not in space
    }
    setup = TuningConfig(
        space=space,
        fixed_config={
            **family.stage1_fixed_config,
            **best_performer,
            "total_timesteps": CONTINUAL_STEPS,
            "exp_name": f"tune_{algo}_continual",
        },
        adapter=family.adapter,
        phase="continual",
        checkpoint_dirs=tuple(spec["checkpoint_dirs"]),
        continual_steps=CONTINUAL_STEPS,
        seeds_per_trial=CONTINUAL_SEEDS_PER_TRIAL,
        pruner_warmup_fraction=0.3,
        eval_episodes=0,
        seed_cap_hours=3.0,
        cap_projection_hours=3.0,
    )
    setup.fixed_config["env_id"] = setup.continual_env_id
    setup.fixed_config.pop("learning_starts", None)

    attrs = scoring_study_attrs(
        seeds_per_trial=setup.seeds_per_trial,
        seed_cap_hours=setup.seed_cap_hours,
        cap_projection_hours=setup.cap_projection_hours,
    )
    for key in (
        "eval_weight",
        "tail_weight",
        "eval_env_id",
        "eval_episodes",
        "eval_seed_start",
        "eval_max_steps",
    ):
        attrs.pop(key)
    attrs.update({
        "phase": setup.phase,
        "algo": algo,
        "deploy_weight": setup.deploy_weight,
        "deploy_window_steps": setup.deploy_window_steps,
        "continual_steps": setup.continual_steps,
        "continual_env_id": setup.continual_env_id,
        "checkpoint_dirs": list(setup.checkpoint_dirs),
        "seed_scheme": seed_scheme(setup.paired_seeds, setup.seeds_per_trial),
    })
    setup.study_attrs = attrs
    setup.study_attrs["objective_revision"] = objective_revision(
        setup.study_attrs, phase="continual"
    )
    return setup
