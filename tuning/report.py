"""
python3 -m tuning.report --journal runs/tuning/stage1.journal --out runs/tuning/report_stage1.html [--stage2-journal runs/tuning/stage2.journal]
"""

from typing import Sequence
import argparse
import html
import os

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import kaleido

    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False

CHECK_COLORS = {
    "pruned": "#999999",
    "checkd_1seed": "#f0ad4e",
}

STAGE2_METRICS = ("robustness", "adaptation", "deploy")


def _is_full(trial) -> bool:
    check = trial.user_attrs.get("check")
    return isinstance(check, str) and check.startswith("full_")


def _check_color(check: str):
    if check in CHECK_COLORS:
        return CHECK_COLORS[check]
    if check.startswith("full_"):
        return "#5cb85c"
    if check.startswith("capped_"):
        return "#d9534f"
    return None


def _load_storage(journal_path: str):
    return JournalStorage(JournalFileBackend(journal_path))


def _studies(storage) -> list[optuna.Study]:
    infos = sorted(optuna.study.get_all_study_names(storage))
    return [optuna.load_study(study_name=n, storage=storage) for n in infos]


def _completed(study: optuna.Study):
    return study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))


def _fig_html(fig, png_dir, name, first):
    parts = [
        fig.to_html(full_html=False, include_plotlyjs=("inline" if first else False))
    ]
    if png_dir is not None and HAS_KALEIDO:
        os.makedirs(png_dir, exist_ok=True)
        try:
            fig.write_image(os.path.join(png_dir, f"{name}.png"))
        except Exception as exc:
            parts.append(f"<p><em>PNG export failed: {html.escape(str(exc))}</em></p>")
    return "".join(parts)


def _safe_plot(fn, note, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        return None, f"{note}: {html.escape(str(exc))}"


def _trial_duration_hours(t):
    if t.datetime_start is None or t.datetime_complete is None:
        return None
    return (t.datetime_complete - t.datetime_start).total_seconds() / 3600.0


def _trial_last_value(t):
    if t.value is not None:
        return t.value
    if t.intermediate_values:
        return t.intermediate_values[max(t.intermediate_values)]
    return None


def _duration_scatter(study):
    by_check: dict[str, list] = {}
    pruned = []
    for t in study.get_trials(deepcopy=False):
        if t.state not in (TrialState.COMPLETE, TrialState.PRUNED):
            continue
        y = _trial_last_value(t)
        dur = _trial_duration_hours(t)
        if y is None or dur is None:
            continue
        if t.state == TrialState.PRUNED:
            pruned.append((dur, y, t.number))
            continue
        check = t.user_attrs.get("check", t.state.name.lower())
        by_check.setdefault(check, []).append((dur, y, t.number))
    if pruned:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.08,
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    for check, pts in sorted(by_check.items()):
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                text=[f"trial {p[2]}" for p in pts],
                mode="markers",
                name=check,
                marker=dict(color=_check_color(check)),
            ),
            row=1,
            col=1,
        )
    if pruned:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pruned],
                y=[p[1] for p in pruned],
                text=[f"trial {p[2]}" for p in pruned],
                mode="markers",
                name="pruned (last reward-rate)",
                marker=dict(color=CHECK_COLORS.get("pruned"), symbol="x"),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="last reward-rate", row=2, col=1)
        fig.update_xaxes(title_text="duration (hours)", row=2, col=1)
    else:
        fig.update_xaxes(title_text="duration (hours)", row=1, col=1)
    fig.update_yaxes(title_text="J", row=1, col=1)
    fig.update_layout(title="Duration vs J")
    return fig


def _seed_consistency_scatter(study):
    xs, ys, text = [], [], []
    for t in _completed(study):
        if not _is_full(t):
            continue
        j1 = t.user_attrs.get("J_seed1")
        jm = t.user_attrs.get("J_mean")
        if j1 is None or jm is None:
            continue
        xs.append(j1)
        ys.append(jm)
        text.append(f"trial {t.number}")
    if not xs:
        return None
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers", text=text))
    fig.update_layout(
        title="Seed consistency: J_seed1 vs J_mean",
        xaxis_title="J_seed1",
        yaxis_title="J_mean",
    )
    return fig


def _seed_keys(trials):
    n = max(
        (
            int(k[len("J_seed") :])
            for t in trials
            for k in t.user_attrs
            if k.startswith("J_seed") and k[len("J_seed") :].isdigit()
        ),
        default=0,
    )
    return [f"J_seed{i + 1}" for i in range(n)]


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sxx * syy) ** 0.5


def selection_diagnostics(study, n_blocks: int = 6) -> dict:
    trials = sorted(
        (t for t in study.get_trials(deepcopy=False) if t.state != TrialState.RUNNING),
        key=lambda t: t.number,
    )
    full = [t for t in trials if _is_full(t)]
    seed_keys = _seed_keys(full)
    unselected = seed_keys[1:]
    out = {
        "n_trials": len(trials),
        "n_full": len(full),
        "n_checkd": sum(
            1 for t in trials if t.user_attrs.get("check") == "checkd_1seed"
        ),
        "n_pruned": sum(1 for t in trials if t.state == TrialState.PRUNED),
        "blocks": [],
        "seed1_offset": None,
        "unselected_pearson": None,
        "unselected_dead_share": None,
    }
    if not trials:
        return out
    size = max(1, -(-len(trials) // n_blocks))
    for start in range(0, len(trials), size):
        block = trials[start : start + size]
        block_full = [t for t in block if _is_full(t)]
        uj = [
            sum(t.user_attrs[k] for k in unselected) / len(unselected)
            for t in block_full
            if unselected and all(k in t.user_attrs for k in unselected)
        ]
        j1 = [t.user_attrs["J_seed1"] for t in block if "J_seed1" in t.user_attrs]
        out["blocks"].append({
            "first_trial": block[0].number,
            "n": len(block),
            "prune_rate": sum(1 for t in block if t.state == TrialState.PRUNED)
            / len(block),
            "seed1_mean": sum(j1) / len(j1) if j1 else None,
            "unselected_mean": sum(uj) / len(uj) if uj else None,
            "n_full": len(block_full),
        })
    scored = [t for t in full if all(k in t.user_attrs for k in seed_keys)]
    if unselected and scored:
        diffs = [
            t.user_attrs["J_seed1"]
            - sum(t.user_attrs[k] for k in unselected) / len(unselected)
            for t in scored
        ]
        out["seed1_offset"] = sum(diffs) / len(diffs)
        values = [t.user_attrs[k] for t in scored for k in unselected]
        out["unselected_dead_share"] = sum(1 for v in values if v < 5.0) / len(values)
    if len(unselected) >= 2 and scored:
        out["unselected_pearson"] = _pearson(
            [t.user_attrs[unselected[-2]] for t in scored],
            [t.user_attrs[unselected[-1]] for t in scored],
        )
    return out


def _selection_diagnostics_html(study):
    d = selection_diagnostics(study)
    fmt = lambda v, spec=".2f": "-" if v is None else format(v, spec)
    head = (
        f"<p>{d['n_trials']} trials: {d['n_full']} fully scored, {d['n_checkd']} checkd, "
        f"{d['n_pruned']} pruned. Seed-1 offset (J_seed1 minus the mean of the "
        f"unselected seeds, over fully scored trials): <b>{fmt(d['seed1_offset'])}</b>; "
        f"pearson between the last two unselected seeds: "
        f"<b>{fmt(d['unselected_pearson'])}</b>; unselected seed-runs with J &lt; 5: "
        f"<b>{fmt(d['unselected_dead_share'], '.0%')}</b>.</p>"
        "<p><em>The offset is the pruner's and check's selection on seed 1, not a "
        "seed-index effect. Pearson near 0 means the ranking among fully scored "
        "trials is seed noise. Read the unselected block means, not the seed-1 "
        "means, for whether later trials are better.</em></p>"
    )
    rows = "".join(
        f"<tr><td>{b['first_trial']}</td><td>{b['n']}</td><td>{b['prune_rate']:.0%}</td>"
        f"<td>{fmt(b['seed1_mean'])}</td><td>{fmt(b['unselected_mean'])}</td>"
        f"<td>{b['n_full']}</td></tr>"
        for b in d["blocks"]
    )
    table = (
        "<table border='1' cellspacing='0' cellpadding='4'>"
        "<tr><th>block from trial</th><th>trials</th><th>prune rate</th>"
        "<th>seed-1 J mean</th><th>unselected J mean</th><th>fully scored</th></tr>"
        f"{rows}</table>"
    )
    return "<h3>Selection diagnostics</h3>" + head + table


def _top10_table(study):
    trials = sorted(
        (t for t in _completed(study) if t.value is not None),
        key=lambda t: t.value,
        reverse=True,
    )[:10]
    if not trials:
        return "<p>No completed trials.</p>"
    attr_cols = ["check", "J_mean", "J_std", "hours_mean"] + _seed_keys(trials)
    param_names = sorted({k for t in trials for k in t.params})[:8]
    header = "".join(f"<th>{html.escape(c)}</th>" for c in attr_cols + param_names)
    rows = []
    for t in trials:
        cells = [f"<td>{t.number}</td><td>{t.value:.4f}</td>"]
        for c in attr_cols:
            v = t.user_attrs.get(c, "")
            cells.append(f"<td>{v if not isinstance(v, float) else f'{v:.4f}'}</td>")
        for p in param_names:
            v = t.params.get(p, "")
            cells.append(f"<td>{v if not isinstance(v, float) else f'{v:.4g}'}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table border='1' cellspacing='0' cellpadding='4'>"
        f"<tr><th>trial</th><th>J (mean - k*std)</th>{header}</tr>{''.join(rows)}</table>"
    )


def _filtered_top20_study(study):
    completed = [t for t in _completed(study) if t.value is not None]
    if len(completed) < 5:
        return study
    completed.sort(key=lambda t: t.value, reverse=True)
    n = max(1, len(completed) // 5)
    sub = optuna.create_study(direction=study.direction)
    for t in completed[:n]:
        sub.add_trial(
            optuna.trial.create_trial(
                params=t.params,
                distributions=t.distributions,
                value=t.value,
                state=TrialState.COMPLETE,
            )
        )
    return sub


def _study_section(study, png_dir, first_fig_flag):
    parts = [f"<h2>{html.escape(study.study_name)}</h2>"]
    notes = []

    fig, err = _safe_plot(
        plot_optimization_history, "optimization history unavailable", study
    )
    if fig is not None:
        parts.append(
            _fig_html(fig, png_dir, f"{study.study_name}_history", first_fig_flag[0])
        )
        first_fig_flag[0] = False
    else:
        notes.append(err)

    top_params = None
    fig, err = _safe_plot(
        plot_param_importances,
        "param importances unavailable (need >=2 completed multi-param trials)",
        study,
    )
    if fig is not None:
        parts.append(
            _fig_html(
                fig, png_dir, f"{study.study_name}_importances", first_fig_flag[0]
            )
        )
        first_fig_flag[0] = False
        try:
            top_params = list(optuna.importance.get_param_importances(study).keys())[:6]
        except Exception:
            top_params = None
    else:
        notes.append(err)

    slice_kwargs = {"params": top_params} if top_params else {}
    fig, err = _safe_plot(plot_slice, "slice plot unavailable", study, **slice_kwargs)
    if fig is None and top_params:
        fig, err = _safe_plot(plot_slice, "slice plot unavailable", study)
    if fig is not None:
        parts.append(
            _fig_html(fig, png_dir, f"{study.study_name}_slice", first_fig_flag[0])
        )
        first_fig_flag[0] = False
    else:
        notes.append(err)

    sub = _filtered_top20_study(study)
    fig, err = _safe_plot(
        plot_parallel_coordinate, "parallel coordinate unavailable", sub
    )
    if fig is not None:
        parts.append(
            "<p><em>Parallel coordinate restricted to top-20% trials by value.</em></p>"
            if sub is not study
            else "<p><em>Parallel coordinate over all trials "
            "(fewer than 5 completed).</em></p>"
        )
        parts.append(
            _fig_html(fig, png_dir, f"{study.study_name}_parallel", first_fig_flag[0])
        )
        first_fig_flag[0] = False
    else:
        notes.append(err)

    parts.append(
        _fig_html(
            _duration_scatter(study),
            png_dir,
            f"{study.study_name}_duration",
            first_fig_flag[0],
        )
    )
    first_fig_flag[0] = False

    seed_fig = _seed_consistency_scatter(study)
    if seed_fig is not None:
        parts.append(
            _fig_html(
                seed_fig,
                png_dir,
                f"{study.study_name}_seedconsistency",
                first_fig_flag[0],
            )
        )
        first_fig_flag[0] = False
    else:
        notes.append(
            "seed-consistency scatter unavailable: no fully-seeded trials with "
            "J_seed1/J_mean"
        )

    parts.append("<h3>Top 10</h3>")
    parts.append(_top10_table(study))
    parts.append(_selection_diagnostics_html(study))

    for n in notes:
        parts.append(f"<p><em>{n}</em></p>")
    return "".join(parts)


def _cross_study_section(studies, png_dir, first_fig_flag):
    if len(studies) < 2:
        return ""
    parts = ["<h2>Cross-study comparison</h2>"]

    fig = go.Figure()
    for study in studies:
        trials = sorted(
            (t for t in _completed(study) if t.value is not None),
            key=lambda t: t.number,
        )
        if not trials:
            continue
        best = (
            float("-inf")
            if study.direction == optuna.study.StudyDirection.MAXIMIZE
            else float("inf")
        )
        xs, ys = [], []
        for t in trials:
            best = (
                max(best, t.value)
                if study.direction == optuna.study.StudyDirection.MAXIMIZE
                else min(best, t.value)
            )
            xs.append(t.number)
            ys.append(best)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=study.study_name))
    fig.update_layout(
        title="Best-so-far vs trial number", xaxis_title="trial", yaxis_title="best J"
    )
    parts.append(_fig_html(fig, png_dir, "cross_best_by_trial", first_fig_flag[0]))
    first_fig_flag[0] = False

    fig = go.Figure()
    for study in studies:
        trials = sorted(
            (
                t
                for t in _completed(study)
                if t.value is not None and t.datetime_start and t.datetime_complete
            ),
            key=lambda t: t.datetime_complete,
        )
        if not trials:
            continue
        best = (
            float("-inf")
            if study.direction == optuna.study.StudyDirection.MAXIMIZE
            else float("inf")
        )
        cum_hours = 0.0
        xs, ys = [], []
        for t in trials:
            cum_hours += (
                t.datetime_complete - t.datetime_start
            ).total_seconds() / 3600.0
            best = (
                max(best, t.value)
                if study.direction == optuna.study.StudyDirection.MAXIMIZE
                else min(best, t.value)
            )
            xs.append(cum_hours)
            ys.append(best)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=study.study_name))
    fig.update_layout(
        title="Best-so-far vs cumulative hours",
        xaxis_title="cumulative hours",
        yaxis_title="best J",
    )
    parts.append(_fig_html(fig, png_dir, "cross_best_by_hours", first_fig_flag[0]))
    first_fig_flag[0] = False

    rows = []
    for study in studies:
        trials = [t for t in _completed(study) if t.value is not None]
        if not trials:
            continue
        best = (
            max(trials, key=lambda t: t.value)
            if study.direction == optuna.study.StudyDirection.MAXIMIZE
            else min(trials, key=lambda t: t.value)
        )
        params = ", ".join(f"{k}={v!r}" for k, v in best.params.items())
        rows.append(
            f"<tr><td>{html.escape(study.study_name)}</td><td>{best.number}</td>"
            f"<td>{best.value:.4f}</td><td>{html.escape(params)}</td></tr>"
        )
    parts.append("<h3>Leaderboard</h3>")
    parts.append(
        "<table border='1' cellspacing='0' cellpadding='4'>"
        "<tr><th>study</th><th>trial</th><th>best value</th><th>params</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    return "".join(parts)


def _stage2_section(stage2_studies, png_dir, first_fig_flag):
    if not stage2_studies:
        return ""
    parts = ["<h2>Stage 2 revalidation</h2>"]
    for study in stage2_studies:
        trials = [t for t in _completed(study)]
        if not trials:
            continue
        by_rank: dict[int, list] = {}
        for t in trials:
            rank = t.params.get("config_rank")
            if rank is None:
                continue
            by_rank.setdefault(rank, []).append(t)
        ranks = sorted(by_rank)
        if not ranks:
            continue
        present = [m for m in STAGE2_METRICS if any(m in t.user_attrs for t in trials)]
        if not present:
            continue
        fig = go.Figure()
        for metric in present:
            means, mins, maxs = [], [], []
            for rank in ranks:
                vals = [
                    t.user_attrs[metric]
                    for t in by_rank[rank]
                    if metric in t.user_attrs
                ]
                if not vals:
                    means.append(None)
                    mins.append(0)
                    maxs.append(0)
                    continue
                m = sum(vals) / len(vals)
                means.append(m)
                mins.append(m - min(vals))
                maxs.append(max(vals) - m)
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=[str(r) for r in ranks],
                    y=means,
                    error_y=dict(
                        type="data", symmetric=False, array=maxs, arrayminus=mins
                    ),
                )
            )
        fig.update_layout(
            title=f"{study.study_name}: {'/'.join(present)} by config_rank",
            barmode="group",
            xaxis_title="config_rank",
            yaxis_title="value",
        )
        parts.append(
            _fig_html(fig, png_dir, f"{study.study_name}_stage2bars", first_fig_flag[0])
        )
        first_fig_flag[0] = False
    return "".join(parts)


def build_report(journal_path: str, out_path: str, stage2_journal: str | None = None):
    storage = _load_storage(journal_path)
    all_studies = _studies(storage)
    stage2_studies = [s for s in all_studies if s.study_name.endswith("_stage2")]
    stage1_studies = [s for s in all_studies if s not in stage2_studies]

    if stage2_journal and stage2_journal != journal_path:
        s2_storage = _load_storage(stage2_journal)
        stage2_studies += [
            s for s in _studies(s2_storage) if s.study_name.endswith("_stage2")
        ]

    png_dir = f"{out_path}_png" if HAS_KALEIDO else None
    first_fig_flag = [True]

    body = ["<h1>Tuning report</h1>"]
    for study in stage1_studies:
        body.append(_study_section(study, png_dir, first_fig_flag))
    body.append(_cross_study_section(stage1_studies, png_dir, first_fig_flag))
    body.append(_stage2_section(stage2_studies, png_dir, first_fig_flag))

    footer = (
        "<footer><p>PNG export: "
        + ("enabled" if HAS_KALEIDO else "skipped (kaleido not installed)")
        + "</p></footer>"
    )
    html_doc = (
        "<html><head><meta charset='utf-8'><title>Tuning report</title></head>"
        f"<body>{''.join(body)}{footer}</body></html>"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html_doc)
    return out_path


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage2-journal", default=None)
    args = parser.parse_args(argv)
    out = build_report(args.journal, args.out, stage2_journal=args.stage2_journal)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
