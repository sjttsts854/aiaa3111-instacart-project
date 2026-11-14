"""Historical-memory interventions with a fixed next-basket estimand.

Candidate selection uses only observed history. Evaluation always takes the full
historical catalog and counts targets omitted by a policy as misses.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


KEYS = ["user_id", "product_id"]


def _check_candidates(frame):
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Candidate keys must be nonnull and unique")


def _positive_integer(value, name, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"Invalid {name}")


def _mask_array(mask, size):
    result = np.asarray(mask)
    if result.ndim != 1 or len(result) != size or result.dtype.kind != "b":
        raise ValueError("Mask must be a row-aligned boolean array")
    return result


def add_window_scores(candidates, history, windows=(3, 5, 10), decay=0.8):
    """Add raw count and decayed-count scores, retaining row order and index.

    History age zero denotes the latest observed order. A scoring window L
    includes ages 0,...,L-1; zero scores do not remove historical candidates.
    Existing purchase_count and decay_score columns are never changed.
    """
    _check_candidates(candidates)
    windows = tuple(dict.fromkeys(windows))
    for window in windows:
        _positive_integer(window, "window")
    if not np.isfinite(decay) or not 0 < decay <= 1:
        raise ValueError("Decay must be in (0, 1]")
    if history[KEYS + ["age"]].isna().any().any() or (history.age < 0).any():
        raise ValueError("History must have nonnull keys and nonnegative ages")
    out = candidates.copy()
    candidate_index = pd.MultiIndex.from_frame(candidates[KEYS])
    # All requested windows are short. Discard irrelevant rows before grouping.
    if not windows:
        return out
    recent = history.loc[history.age < max(windows), KEYS + ["age"]].copy()
    recent["_weight"] = np.power(decay, recent.age.to_numpy())
    for window in windows:
        aggregate = recent.loc[recent.age < window].groupby(KEYS).agg(
            frequency=("age", "size"), decay=("_weight", "sum")
        ).reindex(candidate_index, fill_value=0)
        out[f"frequency_w{window}"] = aggregate.frequency.to_numpy(dtype=np.int64)
        out[f"decay_w{window}"] = aggregate.decay.to_numpy(dtype=float)
    return out


def _row_budgets(frame, budget):
    """Convert a user-indexed Series/dict or row-aligned array to budgets."""
    if budget is None:
        raise ValueError("Matched policies require per-user budgets")
    if isinstance(budget, Mapping):
        budget = pd.Series(budget)
    if isinstance(budget, pd.Series):
        if budget.index.has_duplicates:
            raise ValueError("Budget user IDs must be unique")
        values = frame.user_id.map(budget).to_numpy()
    else:
        values = np.asarray(budget)
        if values.ndim != 1 or len(values) != len(frame):
            raise ValueError("Budget arrays must be aligned to candidate rows")
    try:
        valid = np.isfinite(values).all() and (values >= 0).all() and (values == np.floor(values)).all()
    except TypeError as exc:
        raise ValueError("Budgets must be nonnegative integers for every user") from exc
    if not valid:
        raise ValueError("Budgets must be nonnegative integers for every user")
    budgets = pd.DataFrame({"user_id": frame.user_id.to_numpy(), "budget": values})
    if budgets.groupby("user_id").budget.nunique().gt(1).any():
        raise ValueError("All rows of a user must have the same budget")
    return values


def candidate_mask(frame, kind, window=5, m=None, budget=None):
    """Select candidates using history only; return a positional boolean mask.

    dual = recent products + M old products ordered by frequency, recency, ID.
    The frequency budget control uses that same ordering over all products.
    The recency control orders by recency, frequency, ID. Series/dict budgets
    map user_id to count; array budgets have one entry per original row.
    """
    _check_candidates(frame)
    _positive_integer(window, "window")
    if kind == "all":
        return np.ones(len(frame), dtype=bool)
    if kind not in {"recent", "dual", "pure_frequency_budget", "pure_recency_budget"}:
        raise ValueError(f"Unknown candidate policy: {kind}")
    if not np.isfinite(frame.orders_since_purchase).all() or (frame.orders_since_purchase < 0).any():
        raise ValueError("Recency must be finite and nonnegative")
    recent = frame.orders_since_purchase.to_numpy() < window
    if kind == "recent":
        return recent
    if not np.isfinite(frame.purchase_count).all() or (frame.purchase_count < 0).any():
        raise ValueError("Frequency must be finite and nonnegative")
    work = frame[KEYS + ["purchase_count", "orders_since_purchase"]].copy()
    work["_row"] = np.arange(len(frame))
    if kind == "dual":
        _positive_integer(m, "m", allow_zero=True)
        work = work.loc[~recent]
        limits = None
        selected = recent.copy()
    else:
        limits = _row_budgets(frame, budget)
        selected = np.zeros(len(frame), dtype=bool)
    if kind == "pure_recency_budget":
        columns = ["user_id", "orders_since_purchase", "purchase_count", "product_id"]
        ascending = [True, True, False, True]
    else:
        columns = ["user_id", "purchase_count", "orders_since_purchase", "product_id"]
        ascending = [True, False, True, True]
    work = work.sort_values(columns, ascending=ascending, kind="stable")
    position = work.groupby("user_id").cumcount().to_numpy()
    row = work._row.to_numpy()
    keep = position < (m if kind == "dual" else limits[row])
    selected[row[keep]] = True
    return selected


def evaluate_fixed(frame, scores, mask, k=10):
    """Evaluate a candidate policy against targets in the FULL input frame.

    Never pass a prefiltered frame: full repeat targets and all original users
    define the estimand. Empty-target recall/NDCG/coverage are NaN. Precision
    and F1 are zero when no recommendation is produced. Macro aggregation
    otherwise matches basketlab.metrics.evaluate.
    """
    _check_candidates(frame)
    _positive_integer(k, "k")
    score = np.asarray(scores)
    eligible = _mask_array(mask, len(frame))
    if score.ndim != 1 or len(score) != len(frame) or not np.isfinite(score).all():
        raise ValueError("Scores must be a finite, row-aligned array")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Labels must be binary")
    per = frame.groupby("user_id").agg(
        repeat_targets=("label", "sum"), full_candidates=("label", "size")
    )
    ranked = frame.loc[eligible, KEYS + ["label"]].copy()
    ranked["score"] = score[eligible]
    ranked = ranked.sort_values(["user_id", "score", "product_id"], ascending=[True, False, True])
    ranked["rank"] = ranked.groupby("user_id").cumcount() + 1
    eligible_counts = ranked.groupby("user_id").agg(candidates=("label", "size"), covered_targets=("label", "sum"))
    per = per.join(eligible_counts).fillna({"candidates": 0, "covered_targets": 0})
    top = ranked.loc[ranked["rank"] <= k].copy()
    top["gain"] = top.label / np.log2(top["rank"] + 1)
    per["hits"] = top.groupby("user_id").label.sum().reindex(per.index, fill_value=0)
    per["recommended"] = top.groupby("user_id").size().reindex(per.index, fill_value=0)
    for column in ["repeat_targets", "full_candidates", "candidates", "covered_targets", "hits", "recommended"]:
        per[column] = per[column].astype(np.int64)
    denominator = per.repeat_targets.replace(0, np.nan)
    per["recall"] = per.hits / denominator
    per["precision"] = (per.hits / per.recommended.replace(0, np.nan)).fillna(0.0)
    per["fixed_budget_precision"] = per.hits / k
    per["f1"] = (2 * per.hits / (per.repeat_targets + per.recommended).replace(0, np.nan)).fillna(0.0)
    discounts = np.r_[0.0, np.cumsum(1 / np.log2(np.arange(1, k + 1) + 1))]
    ideal = pd.Series(discounts[np.minimum(k, per.repeat_targets).to_numpy()], index=per.index).replace(0, np.nan)
    per["ndcg"] = top.groupby("user_id").gain.sum().reindex(per.index, fill_value=0) / ideal
    per["coverage"] = per.covered_targets / denominator
    per["oracle_recall"] = np.minimum(k, per.covered_targets) / denominator
    per["candidate_retention"] = per.candidates / per.full_candidates
    metrics = {
        f"recall_at_{k}": float(per.recall.mean()),
        f"ndcg_at_{k}": float(per.ndcg.mean()),
        f"precision_at_{k}": float(per.precision.mean()),
        f"fixed_budget_precision_at_{k}": float(per.fixed_budget_precision.mean()),
        f"f1_at_{k}": float(per.f1.mean()),
        "users": len(per),
        "empty_repeat_targets": int(per.repeat_targets.eq(0).sum()),
        "mean_recommendation_length": float(per.recommended.mean()),
        "mean_candidates": float(per.candidates.mean()),
        "candidate_coverage": float(per.coverage.mean()),
        f"oracle_recall_at_{k}": float(per.oracle_recall.mean()),
        "candidate_retention": float(per.candidate_retention.mean()),
    }
    return metrics, per.reset_index(), top.drop(columns="gain")


def expansion_ledger(frame, small_top, large_top, small_mask, k=10):
    """Decompose a fixed-score pool expansion into new hits and lost old hits.

    Top lists must result from the same score vector/tie rule with the small
    pool contained in the large pool. The caller controls score identity; this
    function checks the observable membership and hit-accounting implications.
    Every original user remains, including users with no eligible products.
    """
    _check_candidates(frame)
    _positive_integer(k, "k")
    eligible = _mask_array(small_mask, len(frame))
    full_keys = pd.MultiIndex.from_frame(frame[KEYS])
    labels = frame.label.to_numpy()
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Labels must be binary")

    def checked_top(top, name):
        _check_candidates(top)
        if top.groupby("user_id").size().gt(k).any():
            raise ValueError(f"{name} exceeds K")
        keys = pd.MultiIndex.from_frame(top[KEYS])
        rows = full_keys.get_indexer(keys)
        if (rows < 0).any():
            raise ValueError(f"{name} contains products outside the full catalog")
        if "label" in top and not np.array_equal(top.label.to_numpy(), labels[rows]):
            raise ValueError(f"{name} labels disagree with the full catalog")
        result = top[KEYS].copy()
        result["label"] = labels[rows]
        result["_row"] = rows
        return result, keys

    small, small_keys = checked_top(small_top, "small_top")
    large, large_keys = checked_top(large_top, "large_top")
    if not eligible[small._row].all():
        raise ValueError("small_top contains a product excluded by the small pool")
    # With a fixed ranking, an old product absent from small top-K cannot enter
    # large top-K: newly admitted products are the only possible arrivals.
    arrivals = ~large_keys.isin(small_keys)
    if eligible[large.loc[arrivals, "_row"]].any():
        raise ValueError("Top lists violate fixed-score candidate expansion")
    departures = ~small_keys.isin(large_keys)
    per = frame.groupby("user_id").label.sum().rename("repeat_targets").to_frame()
    counts = {
        "hits_small": small.groupby("user_id").label.sum(),
        "hits_large": large.groupby("user_id").label.sum(),
        "new_true_hits": large.loc[arrivals].groupby("user_id").label.sum(),
        "lost_old_true_hits": small.loc[departures].groupby("user_id").label.sum(),
        "new_recommendations": large.loc[arrivals].groupby("user_id").size(),
        "displaced_old_recommendations": small.loc[departures].groupby("user_id").size(),
    }
    for column, values in counts.items():
        per[column] = values.reindex(per.index, fill_value=0).astype(np.int64)
    per["delta_hits"] = per.hits_large - per.hits_small
    if not per.delta_hits.eq(per.new_true_hits - per.lost_old_true_hits).all():
        raise AssertionError("Expansion hit ledger does not balance")
    denominator = per.repeat_targets.replace(0, np.nan)
    per["new_true_recall"] = per.new_true_hits / denominator
    per["lost_old_true_recall"] = per.lost_old_true_hits / denominator
    per["delta_recall"] = per.delta_hits / denominator
    return per.reset_index()
