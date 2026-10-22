"""Compute rl-reliability-metrics on the 2empo run set.

Metrics computed (paper naming -> repo class):
  DT  (Dispersion across Time)   -> IqrWithinRuns
  SRT (Short-term Risk across Time) -> LowerCVaROnDiffs
  LRT (Long-term Risk across Time)  -> LowerCVaROnDrawdown
  DR  (Dispersion across Runs)   -> IqrAcrossRuns
  RR  (Risk across Runs)         -> LowerCVaROnAcross

Input: the existing (already 1000-step moving-averaged) "reward" column
logged by RewardTracker in agents/reward.py -- NOT raw_reward. This is a
deliberate choice (see conversation): using the smoothed column keeps all
45 run folders / 20 seeds usable, since most of them never logged
raw_reward. It means DT/SRT/LRT are measuring dispersion/risk of an
already-smoothed curve rather than the true per-step signal, so they will
read lower than if raw_reward had been used everywhere.

For each seed, the mpo_* (phase 1) and continuous_mpo_* (phase 2) runs are
loaded onto one global step axis, mirroring overlay_rewards.py: phase-2 steps
are offset by phase-1's final step. PHASE and STEP_RANGE (below) control
which portion of that global axis is actually analyzed -- e.g. set
PHASE="phase2_only" to look only at the continual-learning segment, and/or
set STEP_RANGE=(lo, hi) to restrict to a specific global step window.
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.stats

RL_RELIABILITY_REPO = "/home/serenaliu/caltech_linc_home/rl-reliability-metrics"
sys.path.insert(0, RL_RELIABILITY_REPO)
from rl_reliability_metrics.metrics import metrics_online  # noqa: E402
from rl_reliability_metrics.metrics import metric_utils as rl_utils  # noqa: E402

RUNS_DIR = "/data2/serenaliu_data/2empo"

# Set to an explicit list of ints to pin the seed set. Set to None to
# auto-discover every seed that has BOTH an mpo_*_seed_N and a
# continuous_mpo_*_seed_N folder in RUNS_DIR (seeds with only one phase
# present -- e.g. an in-progress run, or a folder moved/renamed -- are
# skipped with a warning rather than crashing the whole script).
SEEDS = None


def discover_seeds():
    seed_re = re.compile(r"seed_(\d+)$")

    def seeds_with_prefix(prefix):
        seeds = set()
        for path in glob.glob(f"{RUNS_DIR}/{prefix}_*_seed_*"):
            m = seed_re.search(path)
            if m:
                seeds.add(int(m.group(1)))
        return seeds

    mpo_seeds = seeds_with_prefix("mpo")
    cont_seeds = seeds_with_prefix("continuous_mpo")

    missing_phase2 = sorted(mpo_seeds - cont_seeds)
    missing_phase1 = sorted(cont_seeds - mpo_seeds)
    if missing_phase2:
        print(f"WARNING: seeds with mpo_* but no continuous_mpo_*, skipping: {missing_phase2}")
    if missing_phase1:
        print(f"WARNING: seeds with continuous_mpo_* but no mpo_*, skipping: {missing_phase1}")

    return sorted(mpo_seeds & cont_seeds)

# --- What to analyze ---
# "both"         -> concatenate mpo_* (phase 1) + continuous_mpo_* (phase 2),
#                   global step axis continues across the phase boundary (old default).
# "phase2_only"  -> keep only the continuous_mpo_* (continual-learning) segment.
# "phase1_only"  -> keep only the mpo_* (phase 1) segment.
# In all cases, step numbers stay on the same *global* axis (phase 2's local
# step column is offset by phase 1's final step), so a STEP_RANGE below means
# the same thing regardless of which phase(s) you keep.
PHASE = "phase2_only"

# --- Which global step range to look at ---
# Set to None to use each curve's full available range (after the PHASE
# filter above). Set to (min_step, max_step) to restrict to that window on
# the global step axis before computing any metric.
STEP_RANGE = None  # e.g. (40_000, 120_000)

# Spacing (in steps) between eval points for the across-run metrics (DR, RR).
# Eval points are auto-generated to span the range common to every seed's
# (filtered) curve -- see main().
EVAL_POINT_SPACING = 5_000


def dt_iqr_within_runs_fast(curves, baseline="curve_range"):
    """Fast equivalent of metrics_online.IqrWithinRuns(window_size=None, eval_points=None).

    That class is mathematically: detrend each curve by first-differencing
    (dy/dx), then take the IQR of the differences over a single window
    spanning the *entire* run (since window_size=None => window_size =
    full run length => there is exactly one eval point, the last one).

    We reimplement this directly instead of calling the class, because its
    generic `apply_window_fn` does an O(n^2) Python-level membership test
    ("timepoint in window_timepoints") which is fine for the small windows
    the original paper used, but is far too slow for our ~150-200k point
    dense curves with a single whole-run window. Verified against the
    class on small subsets to confirm identical results.
    """
    values = []
    for curve in curves:
        diff_curve = rl_utils.differences([curve])[0]
        iqr = scipy.stats.iqr(diff_curve[1, :])
        if baseline == "curve_range":
            iqr = iqr / rl_utils.curve_range([curve])[0]
        elif baseline:
            iqr = iqr / baseline
        values.append(iqr)
    return np.array(values)


def load_seed_curve(seed):
    """Load phase-1 (mpo_*) + phase-2 (continuous_mpo_*) for a seed, on one global step axis.

    Returns:
      curve: 2D array [steps, reward], phase 2's steps offset by phase 1's final step
        (so "global step" is comparable across seeds regardless of phase-1 length).
      phase2_start: the first global step belonging to phase 2 (continual learning).
    """
    sim1_dirs = sorted(glob.glob(f"{RUNS_DIR}/mpo_*_seed_{seed}"), reverse=True)
    sim2_dirs = sorted(glob.glob(f"{RUNS_DIR}/continuous_mpo_*_seed_{seed}"), reverse=True)
    if not sim1_dirs or not sim2_dirs:
        raise FileNotFoundError(f"seed {seed}: missing phase 1 or phase 2 run dir")

    sim1_csv = os.path.join(sim1_dirs[0], "SimEmbodiedAnt_average_rewards.csv")
    sim2_csv = os.path.join(sim2_dirs[0], "SimEmbodiedAnt_average_rewards.csv")

    df1 = pd.read_csv(sim1_csv)
    df2 = pd.read_csv(sim2_csv)

    last_step_1 = df1["step"].max()
    phase2_start = df2["step"].min() + last_step_1

    df2 = df2.copy()
    df2["step"] = df2["step"] + last_step_1
    df_all = pd.concat([df1, df2], ignore_index=True)
    df_all = df_all.drop_duplicates(subset="step").sort_values("step")

    steps = df_all["step"].to_numpy(dtype=float)
    reward = df_all["reward"].to_numpy(dtype=float)
    return np.array([steps, reward]), phase2_start


def filter_curve(curve, phase2_start, phase, step_range):
    """Apply the PHASE and STEP_RANGE selection to one seed's global curve."""
    steps = curve[0, :]
    mask = np.ones_like(steps, dtype=bool)

    if phase == "phase2_only":
        mask &= steps >= phase2_start
    elif phase == "phase1_only":
        mask &= steps < phase2_start
    elif phase != "both":
        raise ValueError(f"Unknown PHASE: {phase!r}")

    if step_range is not None:
        lo, hi = step_range
        mask &= (steps >= lo) & (steps <= hi)

    return curve[:, mask]


def main():
    seeds = SEEDS if SEEDS is not None else discover_seeds()
    print(f"PHASE={PHASE!r}  STEP_RANGE={STEP_RANGE}  seeds={seeds}")

    curves = []
    used_seeds = []
    for seed in seeds:
        raw_curve, phase2_start = load_seed_curve(seed)
        curve = filter_curve(raw_curve, phase2_start, PHASE, STEP_RANGE)
        if curve.shape[1] < 2:
            raise ValueError(
                f"seed {seed}: only {curve.shape[1]} points survive the "
                f"PHASE/STEP_RANGE filter (phase2_start={phase2_start:.0f}). "
                "Widen STEP_RANGE or check PHASE.")
        curves.append(curve)
        used_seeds.append(seed)
        print(f"seed {seed:>3}: {curve.shape[1]:>7} points, "
              f"steps [{curve[0].min():.0f}, {curve[0].max():.0f}] "
              f"(phase2 starts at global step {phase2_start:.0f})")

    # Eval points for DR/RR must lie within every (filtered) seed's range.
    common_min = max(curve[0].min() for curve in curves)
    common_max = min(curve[0].max() for curve in curves)
    if common_min > common_max:
        raise ValueError(
            "No step range is common to all seeds after filtering -- "
            f"common_min={common_min:.0f} > common_max={common_max:.0f}.")
    EVAL_POINTS = list(np.arange(common_min, common_max, EVAL_POINT_SPACING)[1:])
    if not EVAL_POINTS:
        EVAL_POINTS = [common_max]
    print(f"Eval points for DR/RR: {EVAL_POINTS[0]:.0f} .. {EVAL_POINTS[-1]:.0f} "
          f"(n={len(EVAL_POINTS)}, spacing={EVAL_POINT_SPACING})")

    # --- DT: Dispersion across Time (IQR within each run, over the whole run) ---
    # (using dt_iqr_within_runs_fast instead of metrics_online.IqrWithinRuns directly --
    # see docstring for why: the generic implementation is O(n^2) for our curve lengths)
    dt_values = dt_iqr_within_runs_fast(curves, baseline="curve_range")

    # --- SRT: Short-term Risk across Time (lower-tail CVaR on step diffs) ---
    srt_metric = metrics_online.LowerCVaROnDiffs(alpha=0.05, baseline="curve_range")
    srt_values = srt_metric(curves)  # one value per seed

    # --- LRT: Long-term Risk across Time (CVaR on drawdown) ---
    # Reporting both tails:
    #   lower tail = the paper's official/canonical LRT metric. Empirically (see
    #     conversation) its "worst 5%" points are dominated by the early climbing
    #     phase (when the curve is still setting new all-time highs), so it mostly
    #     measures how good the early ramp-up was, not later regression.
    #   upper tail = CVaR of the *worst* 5% of drawdowns, i.e. the deepest
    #     sustained declines from a prior peak. Not the paper's default, but far
    #     more diagnostic for spotting severe/sustained collapse (e.g. forgetting
    #     after a continual-learning phase transition that never fully recovers).
    lrt_lower_metric = metrics_online.LowerCVaROnDrawdown(alpha=0.05, baseline="curve_range")
    lrt_lower_values = lrt_lower_metric(curves)  # one value per seed

    lrt_upper_metric = metrics_online.UpperCVaROnDrawdown(alpha=0.05, baseline="curve_range")
    lrt_upper_values = lrt_upper_metric(curves)  # one value per seed

    # --- DR: Dispersion across Runs (IQR across seeds, at each eval point) ---
    dr_metric = metrics_online.IqrAcrossRuns(
        eval_points=EVAL_POINTS, baseline="curve_range")
    dr_values = dr_metric(curves)  # one value per eval point

    # --- RR: Risk across Runs (lower-tail CVaR across seeds, at each eval point) ---
    rr_metric = metrics_online.LowerCVaROnAcross(
        alpha=0.05, eval_points=EVAL_POINTS, baseline="curve_range")
    rr_values = rr_metric(curves)  # one value per eval point

    print("\n=== Per-run metrics (one value per seed) ===")
    per_run_df = pd.DataFrame({
        "seed": used_seeds,
        "DT_IqrWithinRuns": dt_values,
        "SRT_LowerCVaROnDiffs": srt_values,
        "LRT_LowerCVaROnDrawdown": lrt_lower_values,
        "LRT_UpperCVaROnDrawdown": lrt_upper_values,
    })
    print(per_run_df.to_string(index=False))

    print("\n=== Per-eval-point metrics (across the 20 seeds) ===")
    across_run_df = pd.DataFrame({
        "eval_point_step": EVAL_POINTS,
        "DR_IqrAcrossRuns": dr_values,
        "RR_LowerCVaROnAcross": rr_values,
    })
    print(across_run_df.to_string(index=False))

    print("\n=== Summary (median across seeds / eval points) ===")
    print(f"DT  (median IQR-within-run, across time):     {np.median(dt_values):.6g}")
    print(f"SRT (median lower-CVaR on diffs):              {np.median(srt_values):.6g}")
    print(f"LRT (median lower-CVaR on drawdown, official): {np.median(lrt_lower_values):.6g}")
    print(f"LRT (median upper-CVaR on drawdown, diagnostic): {np.median(lrt_upper_values):.6g}")
    print(f"DR  (median IQR-across-runs, across eval pts): {np.median(dr_values):.6g}")
    print(f"RR  (median lower-CVaR-across-runs):           {np.median(rr_values):.6g}")

    out_csv_dir = os.path.join(RUNS_DIR, "reliability_metrics")
    os.makedirs(out_csv_dir, exist_ok=True)
    per_run_df.to_csv(os.path.join(out_csv_dir, "per_run_metrics.csv"), index=False)
    across_run_df.to_csv(os.path.join(out_csv_dir, "across_run_metrics.csv"), index=False)
    print(f"\nSaved results to {out_csv_dir}/")


if __name__ == "__main__":
    main()
