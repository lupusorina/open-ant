import math

import optuna

from .callbacks import MeanAggregator
from .cfg import TuningConfig
from .search_space import sample


def build_objective(setup: TuningConfig):
    def objective(trial: optuna.Trial) -> float:
        config = {k: v for k, v in setup.fixed_config.items() if k != "base_seed"}
        config.update(sample(setup.space, trial))
        config["seed"] = int(setup.fixed_config.get("base_seed", 0)) + trial.number

        total_steps = int(config[setup.total_steps_key])
        agg = MeanAggregator(total_steps, setup.last_fraction)
        state = {"pruned": False, "diverged": False}

        def on_report(step: int, value: float) -> bool:
            if not math.isfinite(value):
                state["diverged"] = True
                return True
            agg.add(step, value)
            trial.report(value, step)
            if trial.should_prune():
                state["pruned"] = True
                return True
            return False

        fallback = setup.trainable(config, setup.env_factory, on_report)

        if state["pruned"]:
            raise optuna.TrialPruned()
        if state["diverged"]:
            return float("nan")

        value = agg.objective()
        if math.isnan(value):
            value = float(fallback)
        return value

    return objective
