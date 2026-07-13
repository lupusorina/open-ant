from dataclasses import dataclass
from typing import Any, Dict

from .search_space import SpaceSpec


@dataclass
class TuningConfig:
    space: SpaceSpec
    trainable: Any
    fixed_config: Dict[str, Any]
    env_factory: Any | None = None
    last_fraction: float = 0.25
    pruner_warmup_fraction: float = 0.3
    total_steps_key: str = "total_timesteps"
