import os

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def create_study(
    name: str,
    storage_dir: str,
    sampler_seed: int | None = None,
    n_startup_trials_sampler: int = 25,
    n_startup_trials_pruner: int = 10,
    pruner_warmup_steps: int = 0,
) -> optuna.Study:
    os.makedirs(storage_dir, exist_ok=True)
    storage = JournalStorage(
        JournalFileBackend(os.path.join(storage_dir, f"{name}.journal"))
    )
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        n_startup_trials=n_startup_trials_sampler,
        seed=sampler_seed,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials_pruner,
        n_warmup_steps=pruner_warmup_steps,
    )
    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction=optuna.study.StudyDirection.MAXIMIZE,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
