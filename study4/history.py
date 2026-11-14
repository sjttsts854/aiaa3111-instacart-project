"""Historical-covariate standardization of descriptive Recall differences.

All four specifications are fixed before confirmation. Future target counts
are used only to identify users for whom Recall is defined, never as strata.
Intervals condition on the observed support, pooled weights and fitted models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = ["3-5", "6-15", "16+"]
CANDIDATE_BINS = [0, 20, 50, 100, 200, 1000000]
HISTORICAL_BASKET_BINS = [0, 5, 10, 15, 25, 1000000]
BINNING = {
    "candidate_only": {"candidate_bins": CANDIDATE_BINS},
    "candidate_historical_basket": {
        "candidate_bins": CANDIDATE_BINS,
        "historical_basket_bins": HISTORICAL_BASKET_BINS,
    },
}
MINIMUMS = [10, 20]
CELL_COLUMNS = [
    "specification", "history_group", "candidate_bin", "historical_basket_bin",
    "candidate_lower", "candidate_upper", "historical_basket_lower",
    "historical_basket_upper", "users", "pooled_weight", "recall",
]


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def make_details(per_user_df, frame):
    """Combine one seed-averaged outcome with full historical candidate counts.

    ``basket_mean`` must be computed from the observed prior orders only. The
    caller is responsible for its provenance; terminal-order fields are ignored.
    """
    needed = ["user_id", "recall", "repeat_targets"]
    missing = set(needed) - set(per_user_df.columns)
    if missing:
        raise ValueError(f"Missing per-user fields: {sorted(missing)}")
    required_frame = {"user_id", "product_id", "history_orders", "basket_mean"}
    missing = required_frame - set(frame.columns)
    if missing:
        raise ValueError(f"Missing historical candidate fields: {sorted(missing)}")
    if frame.empty or per_user_df.empty:
        raise ValueError("Empty analysis input")
    if per_user_df.user_id.isna().any() or frame[["user_id", "product_id"]].isna().any().any():
        raise ValueError("User and product identifiers must not be missing")
    if per_user_df.user_id.duplicated().any():
        raise ValueError("Per-user rows must be unique; average model seeds first")
    if frame.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Duplicate historical candidates")
    if set(per_user_df.user_id) != set(frame.user_id):
        raise ValueError("Outcome and candidate user sets must match exactly")
    grouped = frame.groupby("user_id", sort=True)
    for field in ["history_orders", "basket_mean"]:
        if not grouped[field].nunique(dropna=False).eq(1).all():
            raise ValueError(f"Inconsistent historical {field} within user")
    info = grouped.agg(candidates=("product_id", "size"),
                       history_orders=("history_orders", "first"),
                       basket_mean=("basket_mean", "first"))
    df = per_user_df[needed].merge(info, on="user_id", validate="one_to_one")
    for field in ["history_orders", "basket_mean", "repeat_targets", "recall"]:
        try:
            df[field] = pd.to_numeric(df[field], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
    if not (df.history_orders.between(3, 100) & df.history_orders.mod(1).eq(0)).all():
        raise ValueError("Expected integral history order counts from 3 to 100")
    if not (np.isfinite(df.basket_mean) & df.basket_mean.gt(0)).all():
        raise ValueError("Historical basket_mean must be finite and positive")
    if not (df.repeat_targets.between(0, df.candidates) & df.repeat_targets.mod(1).eq(0)).all():
        raise ValueError("Invalid repeat-target counts")
    nonempty = df.repeat_targets.gt(0)
    if not df.loc[nonempty, "recall"].between(0, 1).all():
        raise ValueError("Nonempty targets require finite Recall in [0, 1]")
    if df.loc[~nonempty, "recall"].notna().any():
        raise ValueError("Empty repeat targets must have undefined Recall")
    df["history_group"] = pd.cut(df.history_orders, [2, 5, 15, 100], labels=GROUPS)
    return df.sort_values("user_id").reset_index(drop=True)


def standardize_spec(details, binning, minimum, *, n_boot=1000, seed=20261115):
    """Return summary, group means and supported cells for one fixed specification.

    The same retained users enter raw-common and standardized means. The latter
    apply the pooled retained-cell distribution to every history group. Future
    target counts affect only the nonempty-target inclusion indicator.
    """
    _positive_integer(minimum, "minimum")
    _positive_integer(n_boot, "n_boot")
    if binning not in BINNING:
        raise ValueError(f"Unknown binning specification: {binning}")
    spec = BINNING[binning]
    label = f"{binning}_min{minimum}"
    df = details.loc[details.repeat_targets.gt(0)].copy()
    df["candidate_bin"] = pd.cut(df.candidates, spec["candidate_bins"])
    use_basket = "historical_basket_bins" in spec
    df["historical_basket_bin"] = (
        pd.cut(df.basket_mean, spec["historical_basket_bins"]) if use_basket else "not_conditioned"
    )
    keys = ["candidate_bin", "historical_basket_bin"]
    if df[keys].isna().any().any():
        raise ValueError("Predeclared historical bins do not cover the observed nonempty data")
    counts = df.groupby(keys + ["history_group"], observed=True).size().unstack("history_group", fill_value=0)
    counts = counts.reindex(columns=GROUPS, fill_value=0)
    eligible = counts.index[(counts >= minimum).all(axis=1)]
    supported = df.loc[pd.MultiIndex.from_frame(df[keys]).isin(eligible)]
    overall = df.groupby("history_group", observed=True).recall.mean().reindex(GROUPS)
    overall_delta = overall["16+"] - overall["3-5"]
    total_n = df.groupby("history_group", observed=True).size().reindex(GROUPS, fill_value=0)
    summary = {
        "specification": label, "binning": binning, "min_per_group_per_cell": minimum,
        "eligible_cells": len(eligible), "nonempty_users": len(df), "retained_users": len(supported),
        "retained_fraction": len(supported) / len(df) if len(df) else None,
        "raw_overall_long_minus_short": _finite(overall_delta),
        "raw_common_support_long_minus_short": None, "standardized_long_minus_short": None,
        "standardized_ci95_low": None, "standardized_ci95_high": None,
        "raw_common_support_ci95_low": None, "raw_common_support_ci95_high": None,
        "sample_restriction_shift": None, "reweighting_shift_within_common_support": None,
        "point_reversal_from_overall": None, "point_reversal_within_common_support": None,
        "standardized_ci_excludes_zero": None,
    }
    if supported.empty:
        rows = [{"specification": label, "history_group": group,
                 "nonempty_users": int(total_n[group]), "retained_users": 0,
                 "retained_fraction": 0. if total_n[group] else None,
                 "raw_overall_recall": _finite(overall[group]),
                 "raw_common_support_recall": None, "standardized_recall": None}
                for group in GROUPS]
        return summary, pd.DataFrame(rows), pd.DataFrame(columns=CELL_COLUMNS)

    pooled = supported.groupby(keys, observed=True).size()
    weights = pooled / pooled.sum()
    rng = np.random.default_rng(seed)
    boots = {group: np.zeros(n_boot) for group in GROUPS}
    raw_boots = {group: np.zeros(n_boot) for group in GROUPS}
    standardized = {group: 0. for group in GROUPS}
    retained_n = supported.groupby("history_group", observed=True).size()
    cell_rows = []
    for (cb, hb), weight in weights.items():
        cell = supported.loc[supported.candidate_bin.eq(cb) & supported.historical_basket_bin.eq(hb)]
        for group in GROUPS:
            values = cell.loc[cell.history_group.eq(group), "recall"].to_numpy(dtype=float)
            if len(values) < minimum:
                raise AssertionError("Common-support construction failed")
            mean = float(values.mean())
            standardized[group] += float(weight) * mean
            draws = np.empty(n_boot)
            for start in range(0, n_boot, 100):
                size = min(100, n_boot - start)
                indices = rng.integers(len(values), size=(size, len(values)))
                draws[start:start + size] = values[indices].mean(axis=1)
            boots[group] += float(weight) * draws
            raw_boots[group] += len(values) / retained_n[group] * draws
            cell_rows.append({
                "specification": label, "history_group": group,
                "candidate_bin": str(cb), "historical_basket_bin": str(hb),
                "candidate_lower": float(cb.left), "candidate_upper": float(cb.right),
                "historical_basket_lower": float(hb.left) if use_basket else None,
                "historical_basket_upper": float(hb.right) if use_basket else None,
                "users": len(values), "pooled_weight": float(weight), "recall": mean,
            })
    raw = supported.groupby("history_group", observed=True).recall.mean()
    raw_delta = float(raw["16+"] - raw["3-5"])
    adjusted_delta = float(standardized["16+"] - standardized["3-5"])
    ci = np.quantile(boots["16+"] - boots["3-5"], [.025, .975])
    raw_ci = np.quantile(raw_boots["16+"] - raw_boots["3-5"], [.025, .975])
    summary.update({
        "raw_common_support_long_minus_short": raw_delta,
        "standardized_long_minus_short": adjusted_delta,
        "standardized_ci95_low": float(ci[0]), "standardized_ci95_high": float(ci[1]),
        "raw_common_support_ci95_low": float(raw_ci[0]), "raw_common_support_ci95_high": float(raw_ci[1]),
        "sample_restriction_shift": _finite(raw_delta - overall_delta),
        "reweighting_shift_within_common_support": adjusted_delta - raw_delta,
        "point_reversal_from_overall": bool(adjusted_delta * overall_delta < 0),
        "point_reversal_within_common_support": bool(adjusted_delta * raw_delta < 0),
        "standardized_ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
    })
    rows = [{"specification": label, "history_group": group,
             "nonempty_users": int(total_n[group]), "retained_users": int(retained_n[group]),
             "retained_fraction": float(retained_n[group] / total_n[group]),
             "raw_overall_recall": float(overall[group]),
             "raw_common_support_recall": float(raw[group]), "standardized_recall": standardized[group]}
            for group in GROUPS]
    return summary, pd.DataFrame(rows), pd.DataFrame(cell_rows, columns=CELL_COLUMNS)


def analyze_observed(per_user_df, frame, outdir, n_boot=1000, *, seed=20261115,
                     dataset_label="study4_confirmation"):
    """Run all four historical-only diagnostics; never select the best result."""
    _positive_integer(n_boot, "n_boot")
    details = make_details(per_user_df, frame)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    raw = details.groupby("history_group", observed=True).agg(
        users=("user_id", "size"), nonempty_users=("recall", "count"), recall=("recall", "mean"),
        candidate_mean=("candidates", "mean"), candidate_median=("candidates", "median"),
        historical_basket_mean=("basket_mean", "mean"))
    raw = raw.reindex(GROUPS)
    raw[["users", "nonempty_users"]] = raw[["users", "nonempty_users"]].fillna(0).astype(int)
    raw["empty_target_users"] = raw.users - raw.nonempty_users
    raw.to_csv(out / "raw_overall.csv")
    summaries, groups, cells = [], [], []
    for binning in BINNING:
        for minimum in MINIMUMS:
            summary, group, cell = standardize_spec(details, binning, minimum, n_boot=n_boot, seed=seed)
            summaries.append(summary)
            groups.append(group)
            if not cell.empty:
                cells.append(cell)
    sensitivity = pd.DataFrame(summaries)
    sensitivity.to_csv(out / "sensitivity.csv", index=False)
    pd.DataFrame([row for group in groups for row in group.to_dict("records")]).to_csv(out / "groups.csv", index=False)
    cell_table = pd.DataFrame([row for cell in cells for row in cell.to_dict("records")], columns=CELL_COLUMNS)
    cell_table.to_csv(out / "cells.csv", index=False)
    evaluable = sensitivity.standardized_long_minus_short.notna()
    values = sensitivity.loc[evaluable, "standardized_long_minus_short"].to_numpy(dtype=float)
    all_four = bool(evaluable.all())
    positive = bool(len(values) and (values > 0).all())
    negative = bool(len(values) and (values < 0).all())
    result = {
        "dataset": dataset_label, "specifications": len(sensitivity),
        "evaluable_specifications": int(evaluable.sum()), "all_four_evaluable": all_four,
        "users": len(details), "nonempty_users": int(details.repeat_targets.gt(0).sum()),
        "empty_target_users": int(details.repeat_targets.eq(0).sum()),
        "same_nonzero_direction_in_all_four": bool(all_four and (positive or negative)),
        "all_four_standardized_positive": bool(all_four and positive),
        "all_four_standardized_ci_lower_positive": bool(all_four and sensitivity.standardized_ci95_low.gt(0).all()),
        "all_four_reversed_from_overall": bool(all_four and sensitivity.point_reversal_from_overall.all()),
        "all_four_reversed_within_common_support": bool(all_four and sensitivity.point_reversal_within_common_support.all()),
        "standardized_delta_min": _finite(values.min()) if len(values) else None,
        "standardized_delta_max": _finite(values.max()) if len(values) else None,
        "retained_user_min": int(sensitivity.retained_users.min()),
        "retained_user_max": int(sensitivity.retained_users.max()),
        "bootstrap_replicates": int(n_boot), "bootstrap_seed": int(seed),
        "conditioning_variables": ["historical_candidate_count", "historical_mean_basket_size"],
        "future_target_size_used_for_strata": False,
        "population": "users with nonempty next-order repeat targets; this is an outcome-based inclusion restriction",
        "interval_scope": "marginal 95%; conditional on observed support, fixed pooled weights, fitted models, and per-cell sample sizes",
        "interpretation": "descriptive historical-covariate standardization; no causal history effect and no preferred specification selection",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    protocol = {"binning": BINNING, "minimum_per_group_per_cell": MINIMUMS,
                "history_groups": GROUPS, "bin_interval": "left-open, right-closed",
                "future_target_size_used_for_strata": False, "bootstrap_replicates": int(n_boot), "seed": int(seed)}
    (out / "specifications.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "method.txt").write_text(
        "Historical-only stratification of observed history-length differences\n\n"
        "Compare 3-5, 6-15 and 16+ observed orders. Average model-seed metrics within users first. "
        "Count all unique historical candidate products and use historical basket_mean computed "
        "from observed prior orders only. Do not use next-order size, repeat-target counts, or "
        "other future outcomes to define matching cells or weights.\n\n"
        "Recall is undefined for users with no next-order repeat targets, so exclude them from "
        "Recall means and report their counts. This still conditions the analysis population on "
        "a nonempty future repeat target; historical-only strata do not make the population "
        "restriction outcome-free. Changing positive target counts cannot change strata.\n\n"
        "Run candidate-count bins alone and candidate-count bins crossed with historical mean "
        "basket-size bins, each with minimum 10 and 20 users in EVERY history group. Report all "
        "four specifications from specifications.json. Bins are left-open and right-closed. "
        "Common support comprises cells meeting the minimum in all three history groups; "
        "the pooled frequency of retained users supplies identical cell weights for each group.\n\n"
        "Report raw means for all nonempty users, raw means on retained common-support users, "
        "and standardized means on those SAME retained users. Per-group counts and retention "
        "fractions expose selection. The sample-restriction shift and within-common-support "
        "reweighting shift sum to the total change in the long-minus-short contrast. Neither "
        "shift is a causal explanation. Missing support yields undefined standardized results.\n\n"
        f"Use {n_boot} bootstrap replicates resampling users independently WITHIN each "
        "history-group-by-historical-covariate cell. Retain support, pooled weights and cell "
        "sizes in every replicate. Report percentile marginal 95% intervals for raw common-support "
        "and standardized long-minus-short contrasts. The same draws yield both contrasts. "
        "Intervals exclude uncertainty from support construction, estimating weights, selecting "
        "bins or retraining models. Four intervals are sensitivity diagnostics, not independent "
        "confirmatory discoveries.\n\n"
        "Coarse bins leave residual within-cell differences. Historical covariates can themselves "
        "be consequences of accumulated history, and unmeasured group differences remain. "
        "This analysis cannot establish a causal effect of more historical orders. Any diagnostic "
        "that additionally conditions on future target size belongs in a separately labelled "
        "retrospective sensitivity analysis.\n",
        encoding="utf-8")
    return result
