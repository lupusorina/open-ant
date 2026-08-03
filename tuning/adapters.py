import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


def _floor_divide_learning_starts(args) -> None:
    args.learning_starts = args.learning_starts // args.num_envs


@dataclass(frozen=True)
class AgentAdapter:
    module_path: str
    agent_attr: str = "MPO"
    save_checkpoint_takes_step: bool = False
    mutate_args: Optional[Callable[[Any], None]] = _floor_divide_learning_starts
    expand: Dict[str, "Expansion"] = field(default_factory=dict)

    def module(self):
        return importlib.import_module(self.module_path)

    def agent_class(self):
        return getattr(self.module(), self.agent_attr)


@dataclass(frozen=True)
class Expansion:
    inputs: tuple
    fn: Callable[[Dict[str, Any]], Any]


def _layer_sizes(width_key: str, depth_key: str) -> Expansion:
    return Expansion(
        inputs=(width_key, depth_key),
        fn=lambda config: [int(config[width_key])] * int(config[depth_key]),
    )


ACME_LAYER_EXPANSIONS = {
    "policy_layer_sizes": _layer_sizes("policy_width", "policy_depth"),
    "critic_layer_sizes": _layer_sizes("critic_width", "critic_depth"),
}


MPO_ACME = AgentAdapter(
    module_path="agents.mpo.mpo_acme",
    expand=ACME_LAYER_EXPANSIONS,
)

MPO_ACME_ENSEMBLE = AgentAdapter(
    module_path="agents.mpo.mpo_acme_ensemble",
    expand=ACME_LAYER_EXPANSIONS,
)


def _floor_divide_learning_starts_dmpo(args) -> None:
    _floor_divide_learning_starts(args)
    args.policy_learning_starts = args.policy_learning_starts // args.num_envs


DMPO_ACME = AgentAdapter(
    module_path="agents.dmpo.dmpo_acme",
    mutate_args=_floor_divide_learning_starts_dmpo,
    expand=ACME_LAYER_EXPANSIONS,
)
