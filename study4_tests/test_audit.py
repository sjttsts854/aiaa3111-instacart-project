"""Hand-computed cases for the independent raw-set auditor."""
import json

import numpy as np
import pandas as pd
import pytest

from study4.audit import independent_evaluate, independent_ledger, policy_pairs, _compare_record


def test_independent_metrics_keep_full_targets_and_zero_eligible_users():
    scores = pd.DataFrame({"user_id": [1, 1, 1, 2, 3], "product_id": [10, 20, 30, 10, 10],
                           "score": [1., 1., .1, .2, .3]})
    pools = {1: {10, 20, 30}, 2: {10}, 3: {10}}
    repeats = {1: {10, 30}, 2: {10}, 3: set()}
    metrics, per, top = independent_evaluate(scores, {(1, 10), (1, 20)}, repeats, pools, 3)
    assert top.product_id.tolist() == [10, 20]
    assert per.loc[1, "repeat_targets"] == 2
    assert per.loc[1, "recall"] == .5
    assert per.loc[1, "precision"] == .5
    assert per.loc[1, "fixed_budget_precision"] == pytest.approx(1 / 3)
    assert per.loc[1, "ndcg"] == pytest.approx(1 / (1 + 1 / np.log2(3)))
    assert per.loc[2, "recommended"] == per.loc[2, "recall"] == per.loc[2, "precision"] == 0
    assert np.isnan(per.loc[3, "recall"])
    assert metrics["users"] == 3 and metrics["recall_at_3"] == .25
    assert metrics["candidate_coverage"] == .25


def test_history_only_policy_ignores_targets_and_matches_per_user_budgets():
    frame = pd.DataFrame({"user_id": [1, 1, 1, 2, 2], "product_id": [10, 20, 30, 10, 20],
                          "purchase_count": [1, 4, 4, 2, 9], "orders_since_purchase": [0, 9, 5, 8, 10],
                          "label": [1, 0, 1, 0, 1]})
    expected = {(1, 10), (1, 30), (2, 20)}
    assert policy_pairs(frame, {"kind": "dual", "window": 5, "m": 1}) == expected
    frame["label"] = 1 - frame.label
    assert policy_pairs(frame, {"kind": "dual", "window": 5, "m": 1}) == expected
    budget = {1: 2, 2: 0}
    assert policy_pairs(frame, {"kind": "pure_frequency_budget"}, budget) == {(1, 20), (1, 30)}
    assert policy_pairs(frame, {"kind": "pure_recency_budget"}, budget) == {(1, 10), (1, 30)}


def test_independent_ledger_counts_new_and_displaced_true_hits():
    small = pd.DataFrame({"user_id": [1, 1], "product_id": [10, 20]})
    large = pd.DataFrame({"user_id": [1, 1], "product_id": [30, 40]})
    repeats = {1: {10, 40}, 2: set()}
    pools = {1: {10, 20, 30, 40, 50}, 2: {10}}
    allowed = {(1, 10), (1, 20), (1, 50)}
    result = independent_ledger(small, large, allowed, repeats, pools)
    assert result.loc[1, "new_true_hits"] == result.loc[1, "lost_old_true_hits"] == 1
    assert result.loc[1, "delta_hits"] == result.loc[1, "delta_recall"] == 0
    assert result.loc[1, "new_recommendations"] == result.loc[1, "displaced_old_recommendations"] == 2
    assert result.loc[1, "new_true_recall"] == .5
    assert result.loc[2, "repeat_targets"] == 0 and np.isnan(result.loc[2, "delta_recall"])
    invalid = pd.DataFrame({"user_id": [1], "product_id": [50]})
    with pytest.raises(AssertionError, match="Score identity"):
        independent_ledger(small, invalid, allowed, repeats, pools)


def test_saved_record_tampering_is_detected(tmp_path):
    scores = pd.DataFrame({"user_id": [1, 1], "product_id": [10, 20], "score": [2., 1.]})
    pools, repeats, allowed = {1: {10, 20}}, {1: {20}}, {(1, 10)}
    metrics, per, top = independent_evaluate(scores, allowed, repeats, pools, 1)
    per.reset_index().to_parquet(tmp_path / "users.parquet", index=False)
    top.to_parquet(tmp_path / "top10.parquet", index=False)
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    _compare_record(tmp_path, scores, allowed, repeats, pools, 1)
    per["repeat_targets"] = 0  # Incorrectly shrinking denominator to the pruned pool.
    per.reset_index().to_parquet(tmp_path / "users.parquet", index=False)
    with pytest.raises(AssertionError):
        _compare_record(tmp_path, scores, allowed, repeats, pools, 1)
