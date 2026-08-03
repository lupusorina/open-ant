"""
python3 -m tuning.runner --entry agents.tune_mpo_any --name mpo_any_search --storage-dir runs/tuning --workers N --n-trials M
"""

from typing import Any, Dict

from agents.dmpo import tune_dmpo_acme
from agents.mpo import tune_mpo_acme, tune_mpo_acme_ensemble
from agents.mpo.tune_mpo_acme import MPO_FIXED_CONFIG, MPO_SEARCH_SPACE
from tuning.cfg import AlgorithmChoice, TuningConfig

ENTRIES = {
    "mpo_acme": tune_mpo_acme,
    "mpo_acme_ensemble": tune_mpo_acme_ensemble,
    "dmpo_acme": tune_dmpo_acme,
}


def _extra(specific: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in specific.items() if shared.get(k) != v}


def _choice(entry) -> AlgorithmChoice:
    setup = entry.get_tuning_setup()
    return AlgorithmChoice(
        adapter=setup.adapter,
        space=_extra(setup.space, MPO_SEARCH_SPACE),
        fixed_config=_extra(setup.fixed_config, MPO_FIXED_CONFIG),
    )


def get_tuning_setup() -> TuningConfig:
    return TuningConfig(
        space=MPO_SEARCH_SPACE,
        fixed_config=MPO_FIXED_CONFIG,
        algorithms={name: _choice(entry) for name, entry in ENTRIES.items()},
    )
