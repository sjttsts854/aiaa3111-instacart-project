"""Descriptive sensitivity analysis of history-length group comparisons.

Six specifications are fixed before new confirmation data are examined. The
bootstrap conditions on the observed common-support cells and pooled weights.
It resamples users within each (history group, candidate bin, target bin) cell;
it neither retrains models nor estimates a causal effect of historical length.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = ["3-5", "6-15", "16+"]
BINNING = {
    "original": {
        "candidate_bins": [0, 20, 50, 100, 200, 1000000],
        "target_bins": [0, 3, 7, 10, 20, 1000000],
    },
    "finer": {
        "candidate_bins": [0, 10, 20, 35, 50, 75, 100, 150, 200, 1000000],
        "target_bins": [0, 2, 4, 6, 10, 15, 25, 1000000],
    },
    "coarser": {
        "candidate_bins": [0, 30, 75, 150, 1000000],
        "target_bins": [0, 5, 10, 20, 1000000],
    },
}
MINIMUMS = [10, 20]


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _quantiles(values):
    return [float(v) for v in np.quantile(values, [.025, .975])]


def make_details(per_user_df, candidate_frame):
    """Validate one per-user outcome and count each full historical candidate pool."""
    needed = {"user_id", "recall", "repeat_targets"}
    if not needed.issubset(per_user_df.columns):
        raise ValueError(f"Missing per-user fields: {needed-set(per_user_df.columns)}")
    if not {"user_id", "history_orders"}.issubset(candidate_frame.columns):
        raise ValueError("Candidate frame needs user_id and history_orders")
    if per_user_df.user_id.duplicated().any():
        raise ValueError("Per-user rows must be unique; average model seeds first")
    if candidate_frame.empty or per_user_df.empty:
        raise ValueError("Empty analysis input")
    if "product_id" in candidate_frame and candidate_frame.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Duplicate historical candidates")
    if set(per_user_df.user_id) != set(candidate_frame.user_id):
        raise ValueError("Outcome and candidate user sets must match exactly")
    grouped = candidate_frame.groupby("user_id")
    if not grouped.history_orders.nunique().eq(1).all():
        raise ValueError("Inconsistent history lengths within user")
    info = grouped.agg(candidates=("user_id", "size"), history_orders=("history_orders", "first"))
    df = per_user_df[list(needed)].merge(info, on="user_id", validate="one_to_one")
    if not df.history_orders.between(3, 100).all():
        raise ValueError("Expected 3 to 100 observed orders")
    if not (df.history_orders % 1 == 0).all():
        raise ValueError("History order counts must be integral")
    if not (df.repeat_targets.between(0, df.candidates) & (df.repeat_targets % 1 == 0)).all():
        raise ValueError("Invalid repeat-target counts")
    nonempty = df.repeat_targets.gt(0)
    if not df.loc[nonempty, "recall"].between(0, 1).all():
        raise ValueError("Nonempty targets require finite Recall in [0, 1]")
    if df.loc[~nonempty, "recall"].notna().any():
        raise ValueError("Empty repeat targets must have undefined Recall")
    df["history_group"] = pd.cut(df.history_orders, [2, 5, 15, 100], labels=GROUPS)
    return df.sort_values("user_id").reset_index(drop=True)


def standardize_spec(details, binning, minimum, *, n_boot=1000, seed=20261114):
    """Return summary, group and cell tables for a single fixed specification."""
    if minimum < 1 or n_boot < 1:
        raise ValueError("minimum and n_boot must be positive")
    spec = BINNING[binning]
    label = f"{binning}_min{minimum}"
    df = details[details.repeat_targets.gt(0)].copy()
    df["candidate_bin"] = pd.cut(df.candidates, spec["candidate_bins"])
    df["target_bin"] = pd.cut(df.repeat_targets, spec["target_bins"])
    if df[["candidate_bin", "target_bin"]].isna().any().any():
        raise ValueError("Predeclared bins do not cover the observed nonempty data")
    keys = ["candidate_bin", "target_bin"]
    counts = df.groupby(keys+["history_group"], observed=True).size().unstack("history_group", fill_value=0)
    counts = counts.reindex(columns=GROUPS, fill_value=0)
    eligible = counts.index[(counts >= minimum).all(axis=1)]
    mask = pd.MultiIndex.from_frame(df[keys]).isin(eligible)
    supported = df[mask]
    overall = df.groupby("history_group", observed=True).recall.mean().reindex(GROUPS)
    overall_delta = overall.iloc[2]-overall.iloc[0]
    total_n = df.groupby("history_group", observed=True).size().reindex(GROUPS, fill_value=0)
    summary = {
        "specification": label, "binning": binning, "min_per_group_per_cell": minimum,
        "eligible_cells": len(eligible), "nonempty_users": len(df), "retained_users": len(supported),
        "retained_fraction": len(supported)/len(df) if len(df) else None,
        "raw_overall_long_minus_short": _finite(overall_delta),
        "raw_common_support_long_minus_short": None, "standardized_long_minus_short": None,
        "standardized_ci95_low": None, "standardized_ci95_high": None,
        "raw_common_support_ci95_low": None, "raw_common_support_ci95_high": None,
        "sample_restriction_shift": None, "reweighting_shift_within_common_support": None,
        "point_reversal_from_overall": None, "point_reversal_within_common_support": None,
        "standardized_ci_excludes_zero": None,
    }
    if supported.empty:
        rows = [{"specification": label, "history_group": group, "nonempty_users": int(total_n[group]),
                 "retained_users": 0, "retained_fraction": 0. if total_n[group] else None,
                 "raw_overall_recall": _finite(overall[group]), "raw_common_support_recall": None,
                 "standardized_recall": None} for group in GROUPS]
        return summary, pd.DataFrame(rows), pd.DataFrame()

    pooled = supported.groupby(keys, observed=True).size()
    weights = pooled/pooled.sum()
    rng = np.random.default_rng(seed)
    boots = {group: np.zeros(n_boot) for group in GROUPS}
    raw_boots = {group: np.zeros(n_boot) for group in GROUPS}
    standardized = {group: 0. for group in GROUPS}
    retained_n = supported.groupby("history_group", observed=True).size()
    cell_rows = []
    # Independent draws within every history-by-size cell. All contrasts use
    # exactly the same support and the same fixed pooled cell weights.
    for (cb, tb), weight in weights.items():
        cell = supported[supported.candidate_bin.eq(cb) & supported.target_bin.eq(tb)]
        for group in GROUPS:
            values = cell.loc[cell.history_group.eq(group), "recall"].to_numpy()
            if len(values) < minimum:
                raise AssertionError("Common-support construction failed")
            mean = float(values.mean())
            standardized[group] += float(weight)*mean
            draws = np.empty(n_boot)
            for start in range(0, n_boot, 100):
                size = min(100, n_boot-start)
                indices = rng.integers(len(values), size=(size, len(values)))
                draws[start:start+size] = values[indices].mean(axis=1)
            boots[group] += float(weight)*draws
            raw_boots[group] += len(values)/retained_n[group]*draws
            cell_rows.append({"specification": label, "history_group": group,
                              "candidate_bin": str(cb), "target_bin": str(tb),
                              "candidate_lower": float(cb.left), "candidate_upper": float(cb.right),
                              "target_lower": float(tb.left), "target_upper": float(tb.right),
                              "users": len(values), "pooled_weight": float(weight), "recall": mean})
    raw = supported.groupby("history_group", observed=True).recall.mean()
    raw_delta = float(raw["16+"]-raw["3-5"])
    adjusted_delta = float(standardized["16+"]-standardized["3-5"])
    ci = _quantiles(boots["16+"]-boots["3-5"])
    raw_ci = _quantiles(raw_boots["16+"]-raw_boots["3-5"])
    summary.update({"raw_common_support_long_minus_short": raw_delta,
                    "standardized_long_minus_short": adjusted_delta,
                    "standardized_ci95_low": ci[0], "standardized_ci95_high": ci[1],
                    "raw_common_support_ci95_low": raw_ci[0], "raw_common_support_ci95_high": raw_ci[1],
                    "sample_restriction_shift": _finite(raw_delta-overall_delta),
                    "reweighting_shift_within_common_support": adjusted_delta-raw_delta,
                    "point_reversal_from_overall": bool(adjusted_delta*overall_delta < 0),
                    "point_reversal_within_common_support": bool(adjusted_delta*raw_delta < 0),
                    "standardized_ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)})
    rows = []
    for group in GROUPS:
        rows.append({"specification": label, "history_group": group, "nonempty_users": int(total_n[group]),
                     "retained_users": int(retained_n[group]), "retained_fraction": retained_n[group]/total_n[group],
                     "raw_overall_recall": float(overall[group]), "raw_common_support_recall": float(raw[group]),
                     "standardized_recall": standardized[group]})
    return summary, pd.DataFrame(rows), pd.DataFrame(cell_rows)


def analyze_history(per_user_df, candidate_frame, outputdir, *, n_boot=1000, seed=20261114,
                    dataset_label="confirmation_v3"):
    """Run all six fixed sensitivity specifications; never select one as best."""
    out = Path(outputdir)
    out.mkdir(parents=True, exist_ok=True)
    details = make_details(per_user_df, candidate_frame)
    raw = details.groupby("history_group", observed=True).agg(
        users=("user_id", "size"), nonempty_users=("recall", "count"), recall=("recall", "mean"),
        candidate_mean=("candidates", "mean"), candidate_median=("candidates", "median"),
        repeat_target_mean=("repeat_targets", "mean"))
    raw.to_csv(out/"raw_overall.csv")
    summaries, groups, cells = [], [], []
    for binning in BINNING:
        for minimum in MINIMUMS:
            summary, group, cell = standardize_spec(details, binning, minimum, n_boot=n_boot, seed=seed)
            summaries.append(summary); groups.append(group)
            if not cell.empty:
                cells.append(cell)
    sensitivity = pd.DataFrame(summaries)
    sensitivity.to_csv(out/"sensitivity.csv", index=False)
    pd.DataFrame([row for group in groups for row in group.to_dict("records")]).to_csv(out/"groups.csv", index=False)
    cell_table = pd.concat(cells, ignore_index=True) if cells else pd.DataFrame(columns=["specification", "history_group", "users"])
    cell_table.to_csv(out/"cells.csv", index=False)
    evaluable = sensitivity.standardized_long_minus_short.notna()
    values = sensitivity.loc[evaluable, "standardized_long_minus_short"].to_numpy()
    all_six = bool(evaluable.all())
    point_positive = bool(len(values) and (values > 0).all())
    point_negative = bool(len(values) and (values < 0).all())
    result = {
        "dataset": dataset_label, "specifications": len(sensitivity), "evaluable_specifications": int(evaluable.sum()),
        "users": len(details), "nonempty_users": int(details.recall.notna().sum()),
        "all_six_evaluable": all_six,
        "same_nonzero_direction_in_all_six": bool(all_six and (point_positive or point_negative)),
        "all_six_standardized_positive": bool(all_six and point_positive),
        "all_six_standardized_ci_lower_positive": bool(all_six and sensitivity.standardized_ci95_low.gt(0).all()),
        "all_six_reversed_from_overall": bool(all_six and sensitivity.point_reversal_from_overall.all()),
        "all_six_reversed_within_common_support": bool(all_six and sensitivity.point_reversal_within_common_support.all()),
        "standardized_delta_min": _finite(values.min()) if len(values) else None,
        "standardized_delta_max": _finite(values.max()) if len(values) else None,
        "retained_user_min": int(sensitivity.retained_users.min()), "retained_user_max": int(sensitivity.retained_users.max()),
        "bootstrap_replicates": n_boot, "bootstrap_seed": seed,
        "interval_scope": "marginal 95%; conditional on observed support, fixed pooled weights, fitted models, and per-cell sample sizes",
        "interpretation": "descriptive robustness only; no causal effect, no selection of a preferred sensitivity specification",
    }
    (out/"summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    protocol = {"binning": BINNING, "minimum_per_group_per_cell": MINIMUMS,
                "history_groups": GROUPS, "bootstrap_replicates": n_boot, "seed": seed}
    (out/"specifications.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out/"method.txt").write_text(
        "History-length sensitivity analysis\n\n"
        "Compare 3-5, 6-15 and 16+ observed orders. Average model-seed metrics within users first. "
        "Exclude empty repeat targets only from Recall analyses, while reporting their counts. "
        "Count every historical candidate row, rather than only scored top-K rows.\n\n"
        "Use the three fixed candidate/target bin definitions in specifications.json, each with "
        "minimum cell counts of 10 and 20 in EVERY history group. Report all six specifications. "
        "Common support is the intersection of cells meeting this condition in all three groups. "
        "The pooled frequency of retained users determines identical cell weights for every group.\n\n"
        "Report raw means for all nonempty users; raw means on the retained common-support users; "
        "and standardized means on those SAME retained users. Counts and retention fractions expose "
        "sample selection separately from the reweighting. The two descriptive shifts add exactly "
        "to the total change in the long-minus-short contrast; neither shift is a causal explanation.\n\n"
        f"For each specification, run {n_boot} bootstrap repetitions resampling users WITHIN each "
        "history-group-by-candidate-bin-by-target-bin cell. Keep support, pooled weights and cell "
        "sample sizes fixed. Use percentile marginal 95% intervals for standardized and common-support "
        "raw long-minus-short contrasts. These intervals do not include uncertainty from finding "
        "the support, estimating pooled weights, choosing bins or retraining models. Six-specification "
        "intervals are sensitivity diagnostics, not six independent confirmatory discoveries.\n\n"
        "Target basket size is outcome information used only for retrospective analysis. Conditioning "
        "on it can introduce selection effects. Coarse bins leave residual within-cell differences. "
        "The result is not a causal history-length effect or a prospective user-matching method. "
        "A direction reversal can support caution about aggregate comparisons; it cannot establish "
        "that additional history causes better recommendations. An old study rerun remains exploratory; "
        "new users with a new fitted training sample provide a limited independent replication.\n",
        encoding="utf-8")
    return result
