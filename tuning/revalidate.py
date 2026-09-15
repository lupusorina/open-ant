"""
python3 -m tuning.revalidate --journal runs/tuning/stage1.journal --study mpo_stage1 --entry agents.mpo.tune_mpo_stage1 --k 5 --seeds 5 --out runs/tuning/stage2_mpo
"""

from typing import Sequence
import argparse
import importlib
import json
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import traceback

import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.distributions import IntDistribution
from optuna.trial import TrialState

from .callbacks import WallClockGuard
from .driver import run_training
from .objective import full_check_name
from .study import create_study

DEFAULT_CONTINUAL_STEPS = 10_000
DEPLOY_WINDOW_STEPS = 2_000

STAGE2_OBJECTIVE_REVISION = (
    "score_seed = 0.5*robustness(pre-adaptation, env B) + 0.5*adaptation"
    "(whole phase B, {continual_steps} steps); "
    "score_rank = mean - {stability_k:g}*std over usable seeds; "
    "seeds shared across ranks"
)


def stage2_name(study_name: str) -> str:
    if study_name.endswith("_stage1"):
        return study_name[: -len("_stage1")] + "_stage2"
    return study_name + "_stage2"


def _algo_from_study(study_name: str) -> str:
    for suffix in ("_stage1", "_stage2"):
        if study_name.endswith(suffix):
            return study_name[: -len(suffix)]
    return study_name


def eval_spec_from(setup) -> dict:
    if setup.eval_episodes <= 0:
        raise SystemExit(
            "stage 2 scores robustness with the frozen eval, but the entry sets "
            f"eval_episodes={setup.eval_episodes}"
        )
    return {
        "n_episodes": setup.eval_episodes,
        "eval_seed_start": setup.eval_seed_start,
        "max_steps": setup.eval_max_steps,
        "env_id": setup.eval_env_id,
    }


def top_trials(study: optuna.Study, k: int, seeds_per_trial: int):
    check = full_check_name(seeds_per_trial)
    eligible = [
        t
        for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if t.value is not None and t.user_attrs.get("check") == check
    ]
    if len(eligible) < k:
        raise SystemExit(
            f"study {study.study_name!r} has {len(eligible)} trials with "
            f"check={check!r}, fewer than the {k} requested. Let stage 1 run "
            "longer, or lower --k deliberately."
        )
    eligible.sort(key=lambda t: t.value, reverse=True)
    return eligible[:k]


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def _std(values):
    if not values:
        return float("nan")
    mean = _mean(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def pair_seed(rank: int, seed_index: int) -> int:
    del rank
    return 50_000 + seed_index


def save_phase_a_weights(weights, run_dir: str, total_steps: int) -> str:
    weights_dir = os.path.join(run_dir, "weights_and_args")
    os.makedirs(weights_dir, exist_ok=True)
    torch.save(weights, os.path.join(weights_dir, f"checkpoint_{total_steps}.pth"))
    return os.path.abspath(weights_dir)


def revalidate_pair(setup, config, rank: int, seed: int, continual_steps: int):
    exp_name = config.get("exp_name", "run")
    run_name_a = f"{exp_name}_reval_rank{rank}_seed{seed}_phase_a"
    eval_spec = eval_spec_from(setup)

    total_a = int(config["total_timesteps"])
    guard_a = WallClockGuard(
        total_a,
        setup.seed_cap_hours,
        setup.cap_projection_hours,
        setup.cap_projection_min_hours,
    )

    def record_a(step, value):
        return guard_a.exceeded(step)

    config_a = {**config, "seed": seed}
    result_a = run_training(
        setup.adapter,
        config_a,
        on_report=record_a,
        report_every_n_steps=setup.report_every_n_steps,
        run_name=run_name_a,
        return_weights=True,
        eval_spec=eval_spec,
    )
    if guard_a.hit:
        return {"seed": seed, "failed": f"phase A: {guard_a.reason()}"}

    weights = result_a.get("weights")
    if weights is None:
        raise RuntimeError(
            f"phase A of {run_name_a} returned no weights; the continual stage "
            "cannot start from this run"
        )
    weights_dir = save_phase_a_weights(
        weights,
        os.path.join(config.get("runs_directory", "runs"), run_name_a),
        total_a,
    )

    evaluation = result_a["eval"]
    if evaluation is None:
        raise RuntimeError(
            f"phase A of {run_name_a} produced no eval; robustness is half the "
            "stage-2 score and cannot be skipped"
        )
    robustness = float(evaluation["reward_per_second"])

    warmup_steps = int(config.get("batch_size", 0))
    config_b = {
        **config,
        "seed": seed,
        "env_id": setup.continual_env_id,
        "total_timesteps": continual_steps,
        "learning_starts": warmup_steps,
    }
    reports_b = []
    guard_b = WallClockGuard(
        continual_steps,
        setup.seed_cap_hours,
        setup.cap_projection_hours,
        setup.cap_projection_min_hours,
    )

    def record_b(step, value):
        reports_b.append((step, value))
        return guard_b.exceeded(step)

    result_b = run_training(
        setup.adapter,
        config_b,
        on_report=record_b,
        report_every_n_steps=setup.report_every_n_steps,
        run_name=f"{exp_name}_reval_rank{rank}_seed{seed}_phase_b",
        initial_weights=weights,
    )
    if guard_b.hit:
        return {"seed": seed, "failed": f"phase B: {guard_b.reason()}"}

    adaptation = _mean([v for _, v in reports_b])
    deploy = _mean([v for s, v in reports_b if s <= DEPLOY_WINDOW_STEPS])
    if adaptation != adaptation:
        return {"seed": seed, "failed": "phase B produced no reports"}

    return {
        "seed": seed,
        "robustness": robustness,
        "adaptation": adaptation,
        "deploy": deploy,
        "score": 0.5 * robustness + 0.5 * adaptation,
        "phase_b_learning_starts": warmup_steps,
        "phase_a_hours": guard_a.elapsed_hours,
        "phase_b_hours": guard_b.elapsed_hours,
        "phase_a_final": float(result_a["final"]),
        "phase_b_final": float(result_b["final"]),
        "phase_a_run_dir": weights_dir,
    }


def _rank_summary(rank: int, trial, seed_results, stability_k: float):
    usable = [r for r in seed_results if "failed" not in r]
    scores = [r["score"] for r in usable]
    summary = {
        "config_rank": rank,
        "source_trial": trial.number,
        "source_value": trial.value,
        "params": dict(trial.params),
        "seeds": seed_results,
        "n_usable_seeds": len(usable),
        "failed_seeds": [
            {"seed": r["seed"], "reason": r["failed"]}
            for r in seed_results
            if "failed" in r
        ],
        "score_mean": _mean(scores),
        "score_std": _std(scores),
        "score": (
            float("-inf") if not scores else _mean(scores) - stability_k * _std(scores)
        ),
        "phase_a_run_dirs": [r["phase_a_run_dir"] for r in usable],
    }
    for metric in ("robustness", "adaptation", "deploy"):
        values = [r[metric] for r in usable if r.get(metric) is not None]
        summary[metric] = {
            "mean": _mean(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


def _load_setup(entry: str, setup_arg: str | None):
    setup_module = importlib.import_module(entry)
    setup = (
        setup_module.get_tuning_setup(setup_arg)
        if setup_arg is not None
        else setup_module.get_tuning_setup()
    )
    assert setup.adapter is not None, "revalidation requires a single-adapter entry"
    return setup


def _config_for(setup, params) -> dict:
    config = {k: v for k, v in setup.fixed_config.items() if k != "base_seed"}
    config.update(params)
    return config


def _worker(entry, setup_arg, params_by_rank, tasks, continual_steps, threads, queue):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    setup = _load_setup(entry, setup_arg)
    for rank, seed_index in tasks:
        seed = pair_seed(rank, seed_index)
        try:
            result = revalidate_pair(
                setup,
                _config_for(setup, params_by_rank[rank]),
                rank,
                seed,
                continual_steps,
            )
        except Exception:
            result = {"seed": seed, "failed": traceback.format_exc()}
        queue.put((rank, seed_index, result))


def _run_grid(
    setup, entry, setup_arg, trials, n_seeds, continual_steps, workers, threads
):
    tasks = [(r, s) for r in range(len(trials)) for s in range(n_seeds)]
    results = [[None] * n_seeds for _ in trials]

    if workers <= 1:
        for rank, seed_index in tasks:
            seed = pair_seed(rank, seed_index)
            print(f"  rank {rank} seed {seed}")
            results[rank][seed_index] = revalidate_pair(
                setup,
                _config_for(setup, trials[rank].params),
                rank,
                seed,
                continual_steps,
            )
        return results

    params_by_rank = [dict(t.params) for t in trials]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for wid in range(workers):
        slice_ = tasks[wid::workers]
        if not slice_:
            continue
        p = ctx.Process(
            target=_worker,
            args=(
                entry,
                setup_arg,
                params_by_rank,
                slice_,
                continual_steps,
                threads,
                queue,
            ),
        )
        p.start()
        procs.append(p)

    received = 0
    while received < len(tasks):
        try:
            rank, seed_index, result = queue.get(timeout=30.0)
        except queue_mod.Empty:
            if all(not p.is_alive() for p in procs):
                break
            continue
        results[rank][seed_index] = result
        received += 1
    for p in procs:
        p.join()

    missing = [
        (r, s)
        for r in range(len(trials))
        for s in range(n_seeds)
        if results[r][s] is None
    ]
    if missing:
        raise RuntimeError(f"workers exited without reporting pairs: {missing}")
    return results


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--setup-arg", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--continual-steps", type=int, default=DEFAULT_CONTINUAL_STEPS)
    parser.add_argument("--storage-dir", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Spawn this many processes over the (rank, seed) grid. 1 runs in-process.",
    )
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    storage_dir = args.storage_dir or os.path.dirname(args.journal) or "."
    journal_name = os.path.splitext(os.path.basename(args.journal))[0]

    setup = _load_setup(args.entry, args.setup_arg)
    eval_spec_from(setup)

    if not os.path.exists(args.journal):
        raise SystemExit(f"journal not found: {args.journal}")
    existing = optuna.study.get_all_study_names(
        JournalStorage(JournalFileBackend(args.journal))
    )
    if args.study not in existing:
        raise SystemExit(
            f"study {args.study!r} not found in {args.journal}; "
            f"existing studies: {sorted(existing)}"
        )

    stage1 = create_study(args.study, storage_dir, journal_name=journal_name)
    trials = top_trials(stage1, args.k, setup.seeds_per_trial)

    print(f"Revalidating top {len(trials)} of {args.study} x {args.seeds} seeds:")
    for rank, trial in enumerate(trials):
        for seed_index in range(args.seeds):
            print(
                f"  rank {rank} (trial {trial.number}, J={trial.value:.4f}) "
                f"seed {pair_seed(rank, seed_index)}"
            )
    if args.dry_run:
        return

    stage2 = create_study(
        stage2_name(args.study),
        storage_dir,
        journal_name=journal_name,
        study_attrs={
            "source_study": args.study,
            "continual_steps": args.continual_steps,
            "continual_env_id": setup.continual_env_id,
            "eval_env_id": setup.eval_env_id,
            "stability_k": setup.stability_k,
            "objective_revision": STAGE2_OBJECTIVE_REVISION.format(
                stability_k=setup.stability_k,
                continual_steps=args.continual_steps,
            ),
        },
    )

    grid = _run_grid(
        setup,
        args.entry,
        args.setup_arg,
        trials,
        args.seeds,
        args.continual_steps,
        args.workers,
        args.threads_per_worker,
    )

    ranking = []
    for rank, trial in enumerate(trials):
        seed_results = grid[rank]
        for seed_index, result in enumerate(seed_results):
            if "failed" in result:
                print(f"  [!] rank {rank} seed {result['seed']}: {result['failed']}")
                continue
            stage2.add_trial(
                optuna.trial.create_trial(
                    params={"config_rank": rank, "seed_index": seed_index},
                    distributions={
                        "config_rank": IntDistribution(0, len(trials) - 1),
                        "seed_index": IntDistribution(0, args.seeds - 1),
                    },
                    value=result["score"],
                    state=TrialState.COMPLETE,
                    user_attrs={
                        **result,
                        "source_trial": trial.number,
                        "source_params": dict(trial.params),
                    },
                )
            )
        ranking.append(_rank_summary(rank, trial, seed_results, setup.stability_k))

    scored = [r for r in ranking if r["n_usable_seeds"]]
    if not scored:
        raise SystemExit("every (config, seed) pair failed; nothing to rank")
    best_performer = max(scored, key=lambda r: r["score"])

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ranking.json"), "w") as f:
        json.dump(
            {
                "study": args.study,
                "stability_k": setup.stability_k,
                "ranking": ranking,
                "best_performer_rank": best_performer["config_rank"],
            },
            f,
            indent=2,
        )
    with open(os.path.join(args.out, "best_performer.json"), "w") as f:
        json.dump(
            {
                "best_performer_config": {
                    **{k: v for k, v in setup.fixed_config.items() if k != "base_seed"},
                    **best_performer["params"],
                },
                "checkpoint_dirs": best_performer["phase_a_run_dirs"],
                "source_trial": best_performer["source_trial"],
                "algo": _algo_from_study(args.study),
            },
            f,
            indent=2,
        )
    print(
        f"Best performer: rank {best_performer['config_rank']} (trial {best_performer['source_trial']}), "
        f"score={best_performer['score']:.4f} "
        f"(mean={best_performer['score_mean']:.4f}, std={best_performer['score_std']:.4f}), "
        f"robustness={best_performer['robustness']['mean']:.4f}, "
        f"adaptation={best_performer['adaptation']['mean']:.4f}"
    )


if __name__ == "__main__":
    main()
