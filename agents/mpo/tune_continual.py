"""
python3 -m tuning.runner --entry agents.mpo.tune_continual --setup-arg best_performer.json --name empo_continual --storage-dir runs/tuning --workers N --n-trials M
"""

from agents.continual import CONTINUAL_STEPS, MPO_CONTINUAL_KEYS, family_for
from agents.continual import get_tuning_setup

CONTINUAL_SPACE_KEYS = MPO_CONTINUAL_KEYS
CONTINUAL_SEARCH_SPACE = family_for("mpo").space()

__all__ = [
    "CONTINUAL_STEPS",
    "CONTINUAL_SPACE_KEYS",
    "CONTINUAL_SEARCH_SPACE",
    "get_tuning_setup",
]
