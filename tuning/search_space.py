from dataclasses import dataclass
from typing import Any, Dict, Sequence, Union


@dataclass(frozen=True)
class Float:
    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class Int:
    low: int
    high: int
    log: bool = False


@dataclass(frozen=True)
class Cat:
    choices: Sequence[Any]


SpaceSpec = Dict[str, Union[Float, Int, Cat]]


def sample(space: SpaceSpec, trial) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, Float):
            out[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
        elif isinstance(spec, Int):
            out[name] = trial.suggest_int(name, spec.low, spec.high, log=spec.log)
        elif isinstance(spec, Cat):
            out[name] = trial.suggest_categorical(name, list(spec.choices))
        else:
            raise TypeError(f"Unknown space for {name!r}: {type(spec).__name__}")
    return out
