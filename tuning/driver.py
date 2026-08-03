import os
from typing import Any, Callable, Dict, Optional

import numpy as np
from skrl.utils import set_seed  # All algo impls currently use skrl wrapper and seeding

from .adapters import AgentAdapter


def build_args(adapter: AgentAdapter, config: Dict[str, Any]):
    module = adapter.module()
    args = module.parse_args([])

    consumed = {k for exp in adapter.expand.values() for k in exp.inputs}
    for key, value in config.items():
        if key in consumed:
            continue
        assert hasattr(args, key), f"{adapter.module_path} has no arg {key!r}"
        setattr(args, key, value)

    for target, exp in adapter.expand.items():
        if all(k in config for k in exp.inputs):
            assert hasattr(args, target), f"{adapter.module_path} has no arg {target!r}"
            setattr(args, target, exp.fn(config))

    return args


def make_envs(module, args, run_name):
    from embodied_ant_env import BackAndForthTask, ForwardTask

    if args.task_type == "forward":
        task = ForwardTask()
    elif args.task_type == "back_and_forth":
        task = BackAndForthTask(
            radius=args.radius_back_and_forth,
            origin=np.array(args.origin_back_and_forth),
        )
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    out = module.make_ant_envs(
        args, task, "", run_name, runs_directory=args.runs_directory
    )
    return out if isinstance(out, tuple) else (None, out)


def run_training(
    adapter: AgentAdapter,
    config: Dict[str, Any],
    metric,
    on_report: Optional[Callable[[int, float], bool]] = None,
    run_name: Optional[str] = None,
) -> float:
    """Train once and return the metric's final value (nan if none)"""
    args = build_args(adapter, config)

    assert not getattr(args, "eval", False), "the driver does not run evaluation"
    assert getattr(args, "weights_path", None) is None, (
        "the driver does not resume from a checkpoint"
    )

    set_seed(args.seed, deterministic=getattr(args, "torch_deterministic", False))
    if adapter.mutate_args is not None:
        adapter.mutate_args(args)

    run_name = run_name or f"{args.exp_name}_seed_{args.seed}"
    os.makedirs(os.path.join(args.runs_directory, run_name), exist_ok=True)

    raw_env, envs = make_envs(adapter.module(), args, run_name)

    agent = adapter.agent_class()(
        args=args,
        envs=envs,
        disk_folder="",
        run_name=run_name,
        runs_directory=args.runs_directory,
    )

    try:
        obs, info = envs.reset()
        agent.initialize_logging(info)

        for _ in range(agent.global_step, args.total_timesteps):
            actions = agent.get_action(obs, False)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            metrics = agent.agent_step(
                next_obs, actions, rewards, terminations, truncations, infos
            )
            step = agent.global_step
            agent.log_step(step, infos, rewards, metrics)
            metric.observe(step, agent, infos, rewards)

            if on_report is not None and step % args.log_every_n_steps == 0:
                value = metric.value(agent)
                if value is not None and on_report(step, value):
                    break

        final = metric.value(agent)
        return float("nan") if final is None else float(final)
    finally:
        agent.cleanup()
        try:
            envs.close()
        finally:
            if raw_env is not None:
                raw_env.close()
