from typing import Any, Optional

import numpy as np


class RewardRateMetric:
    """Reward per second of simulated time (divides each reward by `env_dt` and averages over a fixed)"""

    def observe(self, step: int, agent, infos, rewards) -> None:
        pass

    def value(self, agent) -> Optional[float]:
        tracker = getattr(agent, "reward_tracker", None)
        if tracker is None:
            return None
        avg = tracker.average_reward_per_second
        return None if avg is None else float(avg)


class EpisodicReturnMetric:
    """Moving mean of completed-episode returns"""

    def __init__(self, window: int = 20):
        self.window = window
        self._returns: list[float] = []
        self._running: Optional[np.ndarray] = None

    def observe(self, step: int, agent, infos, rewards) -> None:
        r = np.asarray(_to_numpy(rewards), dtype=np.float64).reshape(-1)
        if self._running is None:
            self._running = np.zeros_like(r)
        self._running += r

        done = _episode_boundaries(infos, r.shape[0])
        if done is None:
            return
        for i in np.flatnonzero(done):
            self._returns.append(float(self._running[i]))
            self._running[i] = 0.0
        del self._returns[: max(0, len(self._returns) - self.window)]

    def value(self, agent) -> Optional[float]:
        if not self._returns:
            return None
        return float(np.mean(self._returns))


def _to_numpy(x: Any) -> Any:
    detach = getattr(x, "detach", None)
    return detach().cpu().numpy() if detach is not None else np.asarray(x)


def _episode_boundaries(infos, n: int) -> Optional[np.ndarray]:
    done = np.zeros(n, dtype=bool)
    found = False
    for key in ("terminated", "truncated", "_final_info"):
        if isinstance(infos, dict) and key in infos:
            done |= np.asarray(_to_numpy(infos[key]), dtype=bool).reshape(-1)
            found = True
    return done if found else None
