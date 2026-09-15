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
    last_fraction: float = 0.5
    pruner_warmup_fraction: float = 0.3
    pruner_percentile: float = 25.0
    total_steps_key: str = "total_timesteps"
    study_attrs: Dict[str, Any] = field(default_factory=dict)
    seeds_per_trial: int = 3
    eval_episodes: int = 20
    eval_seed_start: int = 10_000
    eval_max_steps: int = 1000
    eval_weight: float = 0.7
    tail_weight: float = 0.3
    eval_env_id: str = "SimEmbodiedAnt"
    duration_lambda: float = 0.2
    duration_ref_hours: float = 1.0
    stability_k: float = 0.5
    seed_cap_hours: float = 24.0
    cap_projection_hours: float = 12.0
    cap_projection_min_hours: float = 1.0
    paired_seeds: bool = False
    check_percentile: float = 0.5
    check_min_completed: int = 6
    phase: str = "stage1"
    continual_steps: int = 15_000
    deploy_window_steps: int = 2_000
    checkpoint_dirs: tuple = ()
    deploy_weight: float = 0.5
    continual_env_id: str = "SimEmbodiedAnt"

    def __post_init__(self):
        if bool(self.adapter) == bool(self.algorithms):
            raise ValueError(
                "Either `adapter` or `algorithms` must be set, but not both"
            )
        if self.phase == "continual" and not self.checkpoint_dirs:
            raise ValueError("continual phase requires checkpoint_dirs")
        if not 0.0 <= self.check_percentile <= 1.0:
            raise ValueError("check_percentile must be in [0, 1]")
        if not 0.0 <= self.pruner_percentile <= 100.0:
            raise ValueError("pruner_percentile must be in [0, 100]")
        if not 0.0 <= self.pruner_warmup_fraction <= 1.0:
            raise ValueError("pruner_warmup_fraction must be in [0, 1]")
        if self.check_min_completed < 1:
            raise ValueError(
                "check_min_completed must be at least 1; with 0 there is no population to take a threshold from"
            )
        if self.stability_k < 0.0:
            raise ValueError("stability_k must not be negative")
        if self.cap_projection_hours > self.seed_cap_hours:
            raise ValueError(
                "cap_projection_hours must not exceed seed_cap_hours, or the hard cap would fire before the projection ever could"
            )
