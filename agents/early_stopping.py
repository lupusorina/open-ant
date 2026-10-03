"""Early stopping for Sim1 training, modelled on Stable-Baselines3's
StopTrainingOnNoModelImprovement.

As in SB3, the score is the mean return of deterministic episodes on a separate
eval env (sb3_eval.Evaluator, which uses SB3's evaluate_policy), not the noisy
training-time return.

Every call to check() is one "evaluation". A new best only counts if it beats
the previous best by min_delta_pct (relative), so slow drift inside the noise
doesn't keep training alive forever. After more than max_no_improvement_evals
evaluations in a row without a new best (and only after min_evals evaluations),
check() returns False and writes <run_dir>/converged.json.

The counters are saved to <run_dir>/early_stopping.json on every check, so a
--resume_in_place run continues counting where it left off.

Usage:
    stopper = StopTrainingOnNoImprovement(run_dir, max_no_improvement_evals=3,
                                          min_evals=4, min_delta_pct=0.02)
    ...
    mean_reward, _ = evaluator.evaluate(step)
    if not stopper.check(step, mean_reward):
        break
"""
import os
import json

import numpy as np


class StopTrainingOnNoImprovement:
    def __init__(self, run_dir, max_no_improvement_evals, min_evals=0,
                 min_delta_pct=0.0, verbose=1):
        self.run_dir = run_dir
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.min_delta_pct = min_delta_pct
        self.verbose = verbose

        self.n_calls = 0
        self.best_mean_reward = -np.inf
        self.best_step = None
        self.no_improvement_evals = 0

        self.state_path = os.path.join(run_dir, "early_stopping.json")
        self.converged_path = os.path.join(run_dir, "converged.json")
        self._load()

    def _is_improvement(self, score):
        if not np.isfinite(self.best_mean_reward):
            return True
        # abs() so the threshold still means "better" when returns are negative.
        return score > self.best_mean_reward + self.min_delta_pct * abs(self.best_mean_reward)

    def check(self, step, score):
        """Record one evaluation. Returns False when training should stop."""
        if score is None:  # no finished episode yet
            return True

        self.n_calls += 1
        continue_training = True

        if self._is_improvement(score):
            self.best_mean_reward = float(score)
            self.best_step = int(step)
            self.no_improvement_evals = 0
        elif self.n_calls > self.min_evals:
            self.no_improvement_evals += 1
            if self.no_improvement_evals > self.max_no_improvement_evals:
                continue_training = False

        if self.verbose >= 1:
            print(f"[early stop] step {step}: score {score:.2f}, best {self.best_mean_reward:.2f} "
                  f"@ {self.best_step}, no improvement for {self.no_improvement_evals}/"
                  f"{self.max_no_improvement_evals} evals")

        self._save()
        if not continue_training:
            self._write_converged(step, score)
        return continue_training

    def _state(self):
        return {
            "n_calls": self.n_calls,
            "best_mean_reward": self.best_mean_reward,
            "best_step": self.best_step,
            "no_improvement_evals": self.no_improvement_evals,
        }

    def _save(self):
        with open(self.state_path, "w") as f:
            json.dump(self._state(), f, indent=2)

    def _load(self):
        if not os.path.exists(self.state_path):
            return
        with open(self.state_path) as f:
            state = json.load(f)
        self.n_calls = state["n_calls"]
        self.best_mean_reward = state["best_mean_reward"]
        self.best_step = state["best_step"]
        self.no_improvement_evals = state["no_improvement_evals"]
        print(f"[early stop] resumed state from {self.state_path}: {state}")

    def _write_converged(self, step, score):
        info = dict(self._state(), reason="no_improvement", stop_step=int(step), final_score=float(score),
                    max_no_improvement_evals=self.max_no_improvement_evals,
                    min_evals=self.min_evals, min_delta_pct=self.min_delta_pct)
        with open(self.converged_path, "w") as f:
            json.dump(info, f, indent=2)
        if self.verbose >= 1:
            print(f"[early stop] stopping at step {step}: no new best (>{self.min_delta_pct:.0%}) "
                  f"for {self.no_improvement_evals} evals; best {self.best_mean_reward:.2f} "
                  f"@ step {self.best_step}")


class StopTrainingOnRewardThreshold:
    """SB3's StopTrainingOnRewardThreshold: stop once the best deterministic eval
    return so far (sb3_eval.Evaluator.best_mean_reward) reaches reward_threshold.

    SB3 only runs it after a new best, but the best only changes on a new best, so
    checking it after every eval gives the same result. The best is restored from
    evaluations.csv on --resume_in_place, so no extra state is saved here. Writes
    <run_dir>/converged.json when it fires, like StopTrainingOnNoImprovement.
    """

    def __init__(self, run_dir, reward_threshold, verbose=1):
        self.reward_threshold = reward_threshold
        self.verbose = verbose
        self.converged_path = os.path.join(run_dir, "converged.json")

    def check(self, step, best_mean_reward):
        """Record one evaluation. Returns False when training should stop."""
        continue_training = bool(best_mean_reward < self.reward_threshold)
        if self.verbose >= 1:
            print(f"[early stop] step {step}: best {best_mean_reward:.2f}, "
                  f"threshold {self.reward_threshold:.2f}")
        if not continue_training:
            with open(self.converged_path, "w") as f:
                json.dump({"reason": "reward_threshold", "stop_step": int(step),
                           "best_mean_reward": float(best_mean_reward),
                           "reward_threshold": self.reward_threshold}, f, indent=2)
            if self.verbose >= 1:
                print(f"[early stop] stopping at step {step}: best eval reward "
                      f"{best_mean_reward:.2f} >= threshold {self.reward_threshold}")
        return continue_training
