from tuning.common import (
    STAGE1_DOMAIN_RANDOMIZATION,
    STAGE1_ENV_CONFIG,
    objective_revision,
    scoring_study_attrs,
    seed_scheme,
)
from tuning.search_space import Cat, Float, Int

STAGE1_SEARCH_SPACE = {
    "policy_lr": Float(1e-4, 2.5e-3, log=True),
    "q_lr": Float(2e-4, 2e-3, log=True),
    "gamma": Float(0.88, 0.98),
    "batch_size": Int(128, 320, step=32),
    "use_layer_norm": Cat([True, False]),
    "dual_lr": Float(1e-3, 5e-2, log=True),
    "epsilon_eta": Float(0.3, 2.0, log=True),
    "epsilon_mu_kl": Float(0.01, 0.5, log=True),
    "epsilon_sigma_kl": Float(1e-7, 1e-4, log=True),
    "max_grad_norm": Float(0.15, 20.0, log=True),
    "samples_per_insert": Float(512.0, 2048.0, log=True),
    "sample_action_num": Int(32, 96, step=16),
    "target_policy_update_period": Int(40, 400, step=20),
    "target_critic_update_period": Int(40, 400, step=20),
    "policy_width": Int(48, 128, step=16),
    "policy_depth": Int(4, 5),
    "critic_width": Int(512, 1536, step=256),
    "critic_depth": Int(4, 5),
}

STAGE1_DISTRIBUTIONAL_SPACE = {
    "num_atoms": Cat([51, 101, 151, 201]),
    "vmin": Float(-1000.0, -100.0),
    "vmax": Float(5.0, 100.0, log=True),
}

STAGE1_FIXED_CONFIG = {
    **STAGE1_ENV_CONFIG,
    "td_horizon": 1,
    "log_every_n_steps": 1000,
}

STAGE1_STUDY_ATTRS = scoring_study_attrs()
STAGE1_STUDY_ATTRS.update(
    {
        "domain_randomization": STAGE1_DOMAIN_RANDOMIZATION,
        "seed_scheme": seed_scheme(
            STAGE1_STUDY_ATTRS["paired_seeds"], STAGE1_STUDY_ATTRS["seeds_per_trial"]
        ),
        "space_revision": "stage1-rev4",
        "objective_revision": objective_revision(STAGE1_STUDY_ATTRS),
    }
)
