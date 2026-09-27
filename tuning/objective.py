import math
import traceback

import optuna

from .callbacks import MeanAggregator, WallClockGuard
from .cfg import TuningConfig
from .common import seed_for
from .driver import run_training
from .search_space import sample


def full_check_name(seeds_per_trial: int) -> str:
    return f"full_{seeds_per_trial}seed"


def _statistics(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return mean, std


def _check_threshold(trial: optuna.Trial, percentile: float):
    values = [
        t.user_attrs["J_seed1"]
        for t in trial.study.get_trials(deepcopy=False)
        if t.number != trial.number and "J_seed1" in t.user_attrs
    ]
    if not values:
        return None, 0
    values.sort()
    position = (len(values) - 1) * percentile
    low = int(position)
    high = min(low + 1, len(values) - 1)
    threshold = values[low] + (values[high] - values[low]) * (position - low)
    return threshold, len(values)


def _record_value(trial, setup: TuningConfig, j_seeds, hours_seeds) -> float:
    j_mean, j_std = _statistics(j_seeds)
    trial.set_user_attr("J_mean", j_mean)
    trial.set_user_attr("J_std", j_std)
    trial.set_user_attr("hours_mean", sum(hours_seeds) / len(hours_seeds))
    return j_mean - setup.stability_k * j_std


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

        for target, exp in adapter.expand.items():
            if all(k in config for k in exp.inputs):
                trial.set_user_attr(target, exp.fn(config))

        base_seed = int(setup.fixed_config.get("base_seed", 0))
        total_steps = int(config[setup.total_steps_key])
        continual = setup.phase == "continual"
        eval_spec = None
        if not continual and setup.eval_episodes > 0:
            eval_spec = {
                "n_episodes": setup.eval_episodes,
                "eval_seed_start": setup.eval_seed_start,
                "max_steps": setup.eval_max_steps,
                "env_id": setup.eval_env_id,
            }

        j_seeds = []
        hours_seeds = []
        eval_episodes_terminated = 0
        full_check = full_check_name(setup.seeds_per_trial)

        for seed_index in range(setup.seeds_per_trial):
            config["seed"] = seed_for(
                base_seed,
                trial.number,
                seed_index,
                paired_seeds=setup.paired_seeds,
                seeds_per_trial=setup.seeds_per_trial,
            )
            agg = MeanAggregator(total_steps, setup.last_fraction)
            deploy_values = []
            state = {"pruned": False, "diverged": False, "capped": False}
            guard = WallClockGuard(
                total_steps,
                setup.seed_cap_hours,
                setup.cap_projection_hours,
                setup.cap_projection_min_hours,
            )

            def on_report(step: int, value: float) -> bool:
                if not math.isfinite(value):
                    state["diverged"] = True
                    return True
                agg.add(step, value)
                if step <= setup.deploy_window_steps:
                    deploy_values.append(value)
                trial.report(value, seed_index * total_steps + step)
                if seed_index == 0 and trial.should_prune():
                    state["pruned"] = True
                    return True
                if guard.exceeded(step):
                    state["capped"] = True
                    if guard.projected_hours is not None:
                        state["projected_hours"] = guard.projected_hours
                    return True
                return False

            initial_weights = None
            if continual:
                from .sessions import weights_from_checkpoint

                initial_weights = weights_from_checkpoint(
                    setup.checkpoint_dirs[seed_index % len(setup.checkpoint_dirs)]
                )

            try:
                result = run_training(
                    adapter,
                    config,
                    on_report,
                    report_every_n_steps=setup.report_every_n_steps,
                    run_name=(
                        f"{config.get('exp_name', 'run')}"
                        f"_trial_{trial.number}_seed_{config['seed']}"
                    ),
                    initial_weights=initial_weights,
                    eval_spec=eval_spec,
                )
            except Exception:
                print(f"[!] trial {trial.number} failed:\n{traceback.format_exc()}")
                raise

            if state["capped"]:
                trial.set_user_attr("cap_hit_seed", seed_index + 1)
                if "projected_hours" in state:
                    trial.set_user_attr(
                        f"cap_projected_hours_seed{seed_index + 1}",
                        state["projected_hours"],
                    )
                if seed_index == 0:
                    trial.set_user_attr("check", "pruned")
                    raise optuna.TrialPruned(guard.reason())
                trial.set_user_attr("check", f"capped_{len(j_seeds)}seed")
                return _record_value(trial, setup, j_seeds, hours_seeds)
            if state["pruned"]:
                trial.set_user_attr("check", "pruned")
                raise optuna.TrialPruned()
            if state["diverged"]:
                trial.set_user_attr("check", "pruned")
                raise optuna.TrialPruned("diverged: metric became non-finite")

            tail = agg.objective()
            if math.isnan(tail):
                tail = float(result["final"])
            if math.isnan(tail):
                trial.set_user_attr("check", "pruned")
                raise optuna.TrialPruned("no finite objective value was produced")

            hours = guard.elapsed_hours
            penalty = setup.duration_lambda * hours / setup.duration_ref_hours
            evaluation = result["eval"]
            if continual:
                if not deploy_values:
                    trial.set_user_attr(
                        f"deploy_window_empty_seed{seed_index + 1}", True
                    )
                    print(
                        f"[!] trial {trial.number} seed {seed_index + 1}: "
                        "no reports in the deploy window; falling back to tail"
                    )
                deploy = (
                    sum(deploy_values) / len(deploy_values) if deploy_values else tail
                )
                j_seed = (
                    setup.deploy_weight * deploy
                    + (1.0 - setup.deploy_weight) * tail
                    - penalty
                )
                trial.set_user_attr(f"deploy_seed{seed_index + 1}", deploy)
            elif evaluation is not None:
                eval_sim1 = float(evaluation["reward_per_second"])
                eval_episodes_terminated += int(evaluation["episodes_terminated"])
                j_seed = (
                    setup.eval_weight * eval_sim1 + setup.tail_weight * tail - penalty
                )
                trial.set_user_attr(f"eval_sim1_seed{seed_index + 1}", eval_sim1)
            else:
                j_seed = tail - penalty

            if not math.isfinite(j_seed):
                trial.set_user_attr("check", "pruned")
                raise optuna.TrialPruned("non-finite J_seed")

            j_seeds.append(j_seed)
            hours_seeds.append(hours)
            trial.set_user_attr(f"train_tail_seed{seed_index + 1}", tail)
            trial.set_user_attr(f"J_seed{seed_index + 1}", j_seed)
            trial.set_user_attr(f"hours_seed{seed_index + 1}", hours)
            trial.set_user_attr("n_eval_episodes_terminated", eval_episodes_terminated)

            if seed_index == 0 and setup.seeds_per_trial > 1:
                threshold, n_completed = _check_threshold(trial, setup.check_percentile)
                if n_completed >= setup.check_min_completed and j_seed < threshold:
                    trial.set_user_attr("check", "checked_1seed")
                    return j_seed

        trial.set_user_attr("check", full_check)
        return _record_value(trial, setup, j_seeds, hours_seeds)

    return objective
