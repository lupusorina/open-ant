"""
python3 -m tuning.runner --entry agents.tune_any --name any_search --storage-dir runs/tuning --workers N --n-trials M
"""

from typing import Any, Dict

from agents.mpo import tune_mpo
from agents.sac import tune_sac
from tuning.cfg import AlgorithmChoice, TuningConfig
from tuning.common import SHARED_FIXED_CONFIG, SHARED_SEARCH_SPACE

ENTRIES = {
    "mpo": tune_mpo,
    "sac": tune_sac,
}


def _extra(specific: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in specific.items() if shared.get(k) != v}


def _choice(entry) -> AlgorithmChoice:
    setup = entry.get_tuning_setup()
    return AlgorithmChoice(
        adapter=setup.adapter,
        space=_extra(setup.space, SHARED_SEARCH_SPACE),
        fixed_config=_extra(setup.fixed_config, SHARED_FIXED_CONFIG),
    )


def get_tuning_setup() -> TuningConfig:
    return TuningConfig(
        space=SHARED_SEARCH_SPACE,
        fixed_config=SHARED_FIXED_CONFIG,
        algorithms={name: _choice(entry) for name, entry in ENTRIES.items()},
    )
