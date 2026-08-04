from typing import Sequence
import argparse
import importlib
import math
import multiprocessing as mp
import os
import sys


def run(
    entry: str,
    name: str,
    storage_dir: str,
    workers: int,
    n_trials: int,
    threads_per_worker: int = 4,
    sampler_seed: int | None = None,
):
    per_worker = int(math.ceil(n_trials / workers))
    ctx = mp.get_context("spawn")
    procs = []
    for wid in range(workers):
        p = ctx.Process(
            target=_study_worker,
            args=(
                entry,
                name,
                storage_dir,
                per_worker,
                threads_per_worker,
                sampler_seed,
                wid,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    failed = [p.exitcode for p in procs if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} worker(s) exited non-zero: {failed}")


def _study_worker(
    entry: str,
    name: str,
    storage_dir: str,
    n_trials: int,
    threads_per_worker: int,
    sampler_seed: int,
    worker_id: int,
):
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    setup_module = importlib.import_module(entry)
    setup = setup_module.get_tuning_setup()

    from .objective import build_objective
    from .study import create_study

    total_steps = int(setup.fixed_config[setup.total_steps_key])
    study = create_study(
        name,
        storage_dir,
        sampler_seed=None if sampler_seed is None else sampler_seed + worker_id,
        pruner_warmup_steps=int(setup.pruner_warmup_fraction * total_steps),
    )
    study.optimize(
        build_objective(setup),
        n_trials=n_trials,
        catch=(Exception,),
        callbacks=[StopAfterConsecutiveFailures()],
    )


class StopAfterConsecutiveFailures:
    def __init__(self, limit: int = 5):
        self.limit = limit
        self.consecutive = 0

    def __call__(self, study, trial):
        import optuna

        if trial.state is optuna.trial.TrialState.FAIL:
            self.consecutive += 1
        else:
            self.consecutive = 0
        if self.consecutive >= self.limit:
            raise RuntimeError(f"{self.consecutive} trials failed in a row - stopping")


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--storage-dir", default="runs/tuning")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=400)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--sampler-seed", type=int, default=None)
    args = parser.parse_args(argv)

    run(
        args.entry,
        args.name,
        args.storage_dir,
        args.workers,
        args.n_trials,
        args.threads_per_worker,
        args.sampler_seed,
    )


if __name__ == "__main__":
    main()
