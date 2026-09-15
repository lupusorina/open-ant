import math
import os

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState


class PopulationPercentilePruner(optuna.pruners.BasePruner):
    def __init__(
        self,
        percentile: float = 25.0,
        n_startup_trials: int = 10,
        n_warmup_steps: int = 0,
    ):
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("percentile must be in [0, 100]")
        if n_startup_trials < 1:
            raise ValueError("n_startup_trials must be at least 1")
        self.percentile = percentile
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        step = trial.last_step
        if step is None or step < self.n_warmup_steps:
            return False
        value = trial.intermediate_values[step]
        if math.isnan(value):
            return True
        population = [
            t.intermediate_values[step]
            for t in study.get_trials(
                deepcopy=False,
                states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING),
            )
            if t.number != trial.number and step in t.intermediate_values
        ]
        population = [v for v in population if not math.isnan(v)]
        if len(population) < self.n_startup_trials:
            return False
        if study.direction == optuna.study.StudyDirection.MINIMIZE:
            threshold = _percentile(population, 100.0 - self.percentile)
            return value > threshold
        threshold = _percentile(population, self.percentile)
        return value < threshold


def _percentile(values: list[float], percentile: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * percentile / 100.0
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def create_study(
    name: str,
    storage_dir: str,
    sampler_seed: int | None = None,
    n_startup_trials_sampler: int = 25,
    n_startup_trials_pruner: int = 10,
    pruner_warmup_steps: int = 0,
    pruner_percentile: float = 25.0,
    journal_name: str | None = None,
    metric_names: tuple[str, ...] = ("J",),
    study_attrs: dict | None = None,
) -> optuna.Study:
    os.makedirs(storage_dir, exist_ok=True)
    storage = JournalStorage(
        JournalFileBackend(
            os.path.join(storage_dir, f"{journal_name or name}.journal")
        )
    )
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        n_startup_trials=n_startup_trials_sampler,
        constant_liar=True,
        seed=sampler_seed,
    )
    pruner = PopulationPercentilePruner(
        percentile=pruner_percentile,
        n_startup_trials=n_startup_trials_pruner,
        n_warmup_steps=pruner_warmup_steps,
    )
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction=optuna.study.StudyDirection.MAXIMIZE,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    if hasattr(study, "set_metric_names"):
        study.set_metric_names(list(metric_names))
    _assert_same_revision(study, study_attrs or {})
    for k, v in (study_attrs or {}).items():
        study.set_user_attr(k, v)
    return study


def _assert_same_revision(study: optuna.Study, study_attrs: dict) -> None:
    advice = "Start a new study (or journal) instead of mixing the two."
    for key in ("objective_revision", "space_revision"):
        if key not in study_attrs:
            continue
        stored = study.user_attrs.get(key)
        if stored is None:
            if study.get_trials(deepcopy=False):
                raise RuntimeError(
                    f"study {study.study_name!r} has trials but records no "
                    f"{key}, so it predates it. {advice}"
                )
        elif stored != study_attrs[key]:
            raise RuntimeError(
                f"study {study.study_name!r} was created with {key}={stored!r} "
                f"but this code has {key}={study_attrs[key]!r}. {advice}"
            )
