from ...tuning.cfg import TuningConfig
from ...tuning.search_space import Cat, Float, Int

from .mpo_default import parse_args as mpo_parse_args, train as mpo_train


MPO_SPACE = {
    "policy_lr": Float(1e-5, 3e-3, log=True),
    "q_lr": Float(1e-5, 3e-3, log=True),
    "dual_lr": Float(1e-4, 1e-1, log=True),
    "dual_constraint": Float(1e-3, 1.0, log=True),
    "kl_mean_constraint": Float(1e-4, 1e-1, log=True),
    "kl_var_constraint": Float(1e-6, 1e-2, log=True),
    "tau": Float(1e-3, 5e-2, log=True),
    "gamma": Float(0.85, 0.99),
    "batch_size": Cat((128, 256, 512)),
    "sample_action_num": Cat((16, 32, 64)),
    "td_horizon": Cat((1, 3, 5)),
    "ensemble": Cat((1, 2)),
    "hidden_dim": Cat((128, 256)),
    "n_hidden_layers": Int(2, 4),
    "use_layer_norm": Cat((True, False)),
    "mstep_iteration_num": Int(1, 10),
    "dual_steps": Int(10, 50),
    "utd_ratio": Int(1, 4),
}
FIXED_CONFIG = {
    "total_timesteps": 40_000,
    "learning_starts": 2000,
    "exp_name": "mpo_trial",
    "runs_directory": "runs/tuning_trials",
    "save_every_n_steps": 40_000,
    "log_every_n_steps": 100,
    "base_seed": 1000,  # used by tuning.objective
}
IGNORED_CONFIG_KEYS = {"base_seed"}


def get_tuning_setup():
    def build_args(config):
        args = mpo_parse_args()
        for key, value in config.items():
            if key in IGNORED_CONFIG_KEYS:
                continue
            assert hasattr(args, key), f"unknown MPO arg from config: {key}"
            setattr(args, key, value)
        return args

    def mpo_trainable(config, env_factory, on_report):
        return mpo_train(
            build_args(config),
            env_factory=getattr(env_factory, "_mpo_args_style", None)
            if env_factory
            else None,
            on_report=on_report,
        )

    return TuningConfig(
        space=dict(MPO_SPACE),
        trainable=mpo_trainable,
        fixed_config=dict(FIXED_CONFIG),
        env_factory=None,
        pruner_warmup_fraction=0.3,
        total_steps_key="total_timesteps",
    )
