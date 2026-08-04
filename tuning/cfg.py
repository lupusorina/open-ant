from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .adapters import AgentAdapter
from .search_space import SpaceSpec


@dataclass(frozen=True)
class AlgorithmChoice:
    adapter: AgentAdapter
    space: SpaceSpec = field(default_factory=dict)
    fixed_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningConfig:
    space: SpaceSpec
    fixed_config: Dict[str, Any]
    adapter: Optional[AgentAdapter] = None
    algorithms: Dict[str, AlgorithmChoice] = field(default_factory=dict)
    algorithm_key: str = "algorithm"
    report_every_n_steps: int = 1000
    # Wide enough that the objective still rewards how fast a trial gets there:
    # in the first study most configurations had converged well before the final
    # quarter, which left the objective almost flat across the good ones.
    last_fraction: float = 0.5
    pruner_warmup_fraction: float = 0.3
    total_steps_key: str = "total_timesteps"

    def __post_init__(self):
        if bool(self.adapter) == bool(self.algorithms):
            raise ValueError(
                "Either `adapter` or `algorithms` must be set, but not both"
            )
