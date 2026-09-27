import os
from typing import Any, Callable, Dict, Optional

from skrl.utils import set_seed  # All algo impls currently use skrl seeding

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


def run_training(
    adapter: AgentAdapter,
    config: Dict[str, Any],
    on_report: Optional[Callable[[int, float], bool]] = None,
    report_every_n_steps: int = 1000,
    run_name: Optional[str] = None,
    initial_weights: Optional[Dict[str, Any]] = None,
    eval_spec: Optional[Dict[str, Any]] = None,
    return_weights: bool = False,
) -> Dict[str, Any]:
    """Train once; return the final reward rate and an optional frozen eval."""
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

    session = adapter.session(args, run_name, weights=initial_weights)
    try:
        obs = session.reset()
        stopped_early = False

        for _ in range(session.global_step, args.total_timesteps):
            obs = session.step(obs)
            step = session.global_step

            if on_report is not None and step % report_every_n_steps == 0:
                value = session.reward_rate
                if value is not None and on_report(step, value):
                    stopped_early = True
                    break

        final = session.reward_rate
        evaluation = None
        if eval_spec is not None and not stopped_early and hasattr(session, "evaluate"):
            evaluation = session.evaluate(**eval_spec)
        result = {
            "final": float("nan") if final is None else float(final),
            "eval": evaluation,
        }
        if return_weights and adapter.snapshot is not None:
            result["weights"] = adapter.snapshot(session.agent)
        return result
    finally:
        session.close()
