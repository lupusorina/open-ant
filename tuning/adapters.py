import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .sessions import MpoSession, SacSession


def _floor_divide_learning_starts(args) -> None:
    args.learning_starts = args.learning_starts // args.num_envs


@dataclass(frozen=True)
class AgentAdapter:
    module_path: str
    session_cls: Callable[..., Any]
    mutate_args: Optional[Callable[[Any], None]] = _floor_divide_learning_starts
    expand: Dict[str, "Expansion"] = field(default_factory=dict)

    def module(self):
        return importlib.import_module(self.module_path)

    def session(self, args, run_name):
        return self.session_cls(self.module(), args, run_name)


@dataclass(frozen=True)
class Expansion:
    inputs: tuple
    fn: Callable[[Dict[str, Any]], Any]


def _layer_sizes(width_key: str, depth_key: str) -> Expansion:
    return Expansion(
        inputs=(width_key, depth_key),
        fn=lambda config: [int(config[width_key])] * int(config[depth_key]),
    )


MPO_LAYER_EXPANSIONS = {
    "policy_layer_sizes": _layer_sizes("policy_width", "policy_depth"),
    "critic_layer_sizes": _layer_sizes("critic_width", "critic_depth"),
}


# One adapter covers MPO, its ensemble and DMPO: they are the same module,
# selected through `critic_type` and `ensemble`.
MPO = AgentAdapter(
    module_path="agents.mpo.mpo_acme",
    session_cls=MpoSession,
    expand=MPO_LAYER_EXPANSIONS,
)

SAC = AgentAdapter(
    module_path="agents.sac.sac_cleanrl",
    session_cls=SacSession,
)
