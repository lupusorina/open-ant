import math
import traceback

import optuna

from .callbacks import MeanAggregator
from .cfg import TuningConfig
from .driver import run_training
from .search_space import sample


def build_objective(setup: TuningConfig):
    def objective(trial: optuna.Trial) -> float:
        config = {k: v for k, v in setup.fixed_config.items() if k != "base_seed"}
        config.update(sample(setup.space, trial))

        adapter = setup.adapter
        if setup.algorithms:
            name = trial.suggest_categorical(
                setup.algorithm_key, sorted(setup.algorithms)
            )
            choice = setup.algorithms[name]
            adapter = choice.adapter
            config.update(choice.fixed_config)
            config.update(sample(choice.space, trial))

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

        try:
            fallback = run_training(
                adapter,
                config,
                setup.metric_factory(),
                on_report,
                run_name=f"trial_{trial.number}_seed_{config['seed']}",
            )
        except Exception:
            print(f"[!] trial {trial.number} failed:\n{traceback.format_exc()}")
            raise

        if state["pruned"]:
            raise optuna.TrialPruned()
        if state["diverged"]:
            raise optuna.TrialPruned("diverged: metric became non-finite")

        value = agg.objective()
        if math.isnan(value):
            value = float(fallback)
        if math.isnan(value):
            raise optuna.TrialPruned("no finite objective value was produced")
        return value

    return objective
