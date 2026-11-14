"""Keep base top-K coverage counts while reranking inside each coverage group.

The returned values are *ranking scores*, not probabilities. They are specific
to K: recompute them when evaluating a different recommendation length.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


def quota_scores(
    frame: pd.DataFrame,
    base_scores,
    rule_scores,
    k: int = 10,
    weight: float = 1.0,
    covered_col: str = "co_narrow_covered",
) -> np.ndarray:
    """Return scores aligned with ``frame`` that preserve base coverage quotas.

    For each user, count the covered/uncovered products in the base top-K.
    Within each group, choose exactly that many products using
    ``(1-weight)*base_scores + weight*rule_scores``. Rank the chosen products
    first and the excluded products second, each by the mixed score. Every tie
    is broken by increasing product_id, matching ``basketlab.metrics.evaluate``.

    Only user_id, product_id and the binary coverage flag are read from frame.
    A coverage flag must be computed exclusively from available history and
    training-fitted rules. Scores must come from a model trained independently
    of the user's held-out target. This function never reads target labels.

    All candidates are retained. Users with fewer than K candidates retain
    them all; users with no/all covered candidates reduce to the mixed ranking.
    At weight=0 the original base score array is returned exactly (copied).
    """
    required = ["user_id", "product_id", covered_col]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Missing candidate columns: {missing}")
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be finite and between 0 and 1")
    if frame[required].isna().any().any():
        raise ValueError("Candidate identifiers and coverage cannot be missing")
    if frame.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Candidate user-product pairs must be unique")
    if not frame[covered_col].isin([0, 1, False, True]).all():
        raise ValueError("Coverage must be binary")

    base = np.asarray(base_scores, dtype=np.float64)
    rule = np.asarray(rule_scores, dtype=np.float64)
    for name, values in [("base_scores", base), ("rule_scores", rule)]:
        if values.ndim != 1 or len(values) != len(frame) or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite vector aligned with candidates")
    if weight == 0 or frame.empty:
        return base.copy()

    ranked = frame[required].reset_index(drop=True).rename(columns={covered_col: "_covered"})
    ranked["_covered"] = ranked._covered.astype(np.int8)
    ranked["_row"] = np.arange(len(ranked))
    ranked["_base"] = base
    ranked["_mixed"] = (1 - weight) * base + weight * rule
    if not np.isfinite(ranked._mixed).all():
        raise ValueError("Mixed scores are not finite")

    ranked = ranked.sort_values(
        ["user_id", "_base", "product_id"], ascending=[True, False, True]
    )
    ranked["_in_base_top"] = ranked.groupby("user_id", sort=False).cumcount().lt(k)
    ranked["_quota"] = ranked.groupby(["user_id", "_covered"], sort=False)._in_base_top.transform("sum")

    ranked = ranked.sort_values(
        ["user_id", "_covered", "_mixed", "product_id"], ascending=[True, True, False, True]
    )
    ranked["_chosen"] = ranked.groupby(["user_id", "_covered"], sort=False).cumcount().lt(ranked._quota)
    ranked = ranked.sort_values(
        ["user_id", "_chosen", "_mixed", "product_id"], ascending=[True, False, False, True]
    )
    result = np.empty(len(ranked), dtype=np.float64)
    result[ranked._row.to_numpy()] = -ranked.groupby("user_id", sort=False).cumcount().to_numpy(dtype=float)
    return result
