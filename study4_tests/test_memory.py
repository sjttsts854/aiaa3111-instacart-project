"""Independent small examples guard the fixed-denominator research design."""
import numpy as np
import pandas as pd
import pytest

from basketlab.metrics import evaluate
from study4.memory import add_window_scores, candidate_mask, evaluate_fixed, expansion_ledger


@pytest.fixture
def catalog():
    # Arbitrary row/index order catches accidental index-aligned assignment.
    return pd.DataFrame({
        "user_id": [2, 1, 1, 2, 1, 3, 1, 2],
        "product_id": [30, 20, 10, 10, 40, 10, 30, 20],
        "purchase_count": [4, 3, 3, 2, 1, 1, 3, 4],
        "orders_since_purchase": [8, 6, 6, 0, 0, 8, 9, 8],
        "label": [0, 1, 0, 1, 1, 0, 1, 0],
    }, index=[70, 30, 60, 10, 90, 20, 80, 40])


def selected(frame, mask, user):
    return set(frame.loc[mask & frame.user_id.eq(user), "product_id"])


def test_scoring_windows_preserve_candidates_and_history_scores():
    candidates = pd.DataFrame({"user_id": [1, 2, 1], "product_id": [20, 10, 10],
                               "purchase_count": [2, 1, 3], "decay_score": [9., 8., 7.]}, index=[8, 4, 9])
    history = pd.DataFrame({"user_id": [1, 1, 1, 1, 1, 2], "product_id": [10, 10, 10, 20, 20, 10],
                            "age": [0, 2, 5, 3, 8, 7]})
    result = add_window_scores(candidates, history, windows=(3, 5), decay=0.8)
    pd.testing.assert_frame_equal(result[candidates.columns], candidates)
    np.testing.assert_array_equal(result.frequency_w3, [0, 0, 2])
    np.testing.assert_array_equal(result.frequency_w5, [1, 0, 2])
    np.testing.assert_allclose(result.decay_w3, [0, 0, 1 + 0.8**2])
    np.testing.assert_allclose(result.decay_w5, [0.8**3, 0, 1 + 0.8**2])


def test_all_pool_matches_frozen_evaluator(catalog):
    scores = np.array([2, 4, 2, 1, 3, 2, 4, 2], dtype=float)
    expected_metrics, expected_per, expected_top = evaluate(catalog, scores, k=2)
    metrics, per, top = evaluate_fixed(catalog, scores, np.ones(len(catalog), dtype=bool), k=2)
    for key, expected in expected_metrics.items():
        assert metrics[key] == pytest.approx(expected, nan_ok=True)
    pd.testing.assert_frame_equal(per[expected_per.columns], expected_per, check_dtype=False)
    pd.testing.assert_frame_equal(top, expected_top)
    np.testing.assert_allclose(per.coverage[:2], 1)
    assert np.isnan(per.coverage.iloc[2])


def test_pruned_pool_has_fixed_targets_and_keeps_empty_users(catalog):
    mask = candidate_mask(catalog, "recent")
    metrics, per, top = evaluate_fixed(catalog, np.ones(len(catalog)), mask, k=2)
    p = per.set_index("user_id")
    assert list(p.index) == [1, 2, 3]
    assert p.loc[1, "repeat_targets"] == 3  # two excluded truths remain misses
    assert p.loc[1, "hits"] == p.loc[1, "covered_targets"] == 1
    assert p.loc[1, "recall"] == p.loc[1, "coverage"] == pytest.approx(1 / 3)
    assert p.loc[1, "precision"] == 1
    assert p.loc[1, "fixed_budget_precision"] == 0.5
    assert p.loc[1, "ndcg"] == pytest.approx(1 / (1 + 1 / np.log2(3)))
    assert p.loc[1, "oracle_recall"] == pytest.approx(1 / 3)
    assert p.loc[3, "candidates"] == p.loc[3, "hits"] == p.loc[3, "recommended"] == 0
    assert p.loc[3, "precision"] == p.loc[3, "f1"] == 0
    assert np.isnan(p.loc[3, "recall"])
    assert metrics["recall_at_2"] == pytest.approx(2 / 3)
    assert len(top) == 2


def test_entirely_empty_eligible_pool(catalog):
    metrics, per, top = evaluate_fixed(catalog, np.ones(len(catalog)), np.zeros(len(catalog), dtype=bool))
    assert top.empty
    assert metrics["users"] == 3
    assert metrics["recall_at_10"] == metrics["precision_at_10"] == metrics["ndcg_at_10"] == 0
    assert per.recommended.eq(0).all() and per.hits.eq(0).all()
    assert per.loc[per.repeat_targets.gt(0), "oracle_recall"].eq(0).all()


def test_dual_budget_and_stable_history_only_ties(catalog):
    dual = candidate_mask(catalog, "dual", m=1)
    assert selected(catalog, dual, 1) == {10, 40}  # tied old 10 wins by ID, 30 loses by recency
    assert selected(catalog, dual, 2) == {10, 20}
    assert selected(catalog, dual, 3) == {10}
    budget = pd.Series(dual.astype(int)).groupby(catalog.user_id.to_numpy()).sum()
    frequency = candidate_mask(catalog, "pure_frequency_budget", budget=budget)
    recency = candidate_mask(catalog, "pure_recency_budget", budget=budget)
    assert selected(catalog, frequency, 1) == {10, 20}
    assert selected(catalog, recency, 1) == {10, 40}
    for mask in (frequency, recency):
        np.testing.assert_array_equal(pd.Series(mask.astype(int)).groupby(catalog.user_id.to_numpy()).sum(), budget)
    changed = catalog.assign(label=1 - catalog.label)
    for kind, kwargs in [("all", {}), ("recent", {}), ("dual", {"m": 1}),
                         ("pure_frequency_budget", {"budget": budget}), ("pure_recency_budget", {"budget": budget})]:
        np.testing.assert_array_equal(candidate_mask(catalog, kind, **kwargs), candidate_mask(changed, kind, **kwargs))
        shuffled = catalog.sample(frac=1, random_state=4)
        chosen = shuffled.loc[candidate_mask(shuffled, kind, **kwargs), ["user_id", "product_id"]]
        expected = catalog.loc[candidate_mask(catalog, kind, **kwargs), ["user_id", "product_id"]]
        assert set(map(tuple, chosen.to_numpy())) == set(map(tuple, expected.to_numpy()))
    np.testing.assert_array_equal(candidate_mask(catalog, "dual", m=0), candidate_mask(catalog, "recent"))


def test_budget_array_and_validation(catalog):
    budgets = catalog.user_id.map({1: 2, 2: 0, 3: 8}).to_numpy()
    mask = candidate_mask(catalog, "pure_frequency_budget", budget=budgets)
    assert mask.sum() == 3
    assert selected(catalog, mask, 2) == set()
    with pytest.raises(ValueError, match="same budget"):
        candidate_mask(catalog, "pure_frequency_budget", budget=np.arange(len(catalog)))
    with pytest.raises(ValueError, match="every user"):
        candidate_mask(catalog, "pure_frequency_budget", budget={1: 2})
    with pytest.raises(ValueError, match="integer"):
        candidate_mask(catalog, "dual", m=1.5)
    with pytest.raises(ValueError, match="boolean"):
        evaluate_fixed(catalog, np.ones(len(catalog)), np.ones(len(catalog)))


def test_expansion_ledger_separates_recovered_and_displaced_targets():
    frame = pd.DataFrame({"user_id": [1]*5 + [2]*3 + [3], "product_id": [1, 2, 3, 4, 5, 1, 2, 3, 1],
                          "label": [1, 0, 1, 0, 1, 1, 0, 1, 0]})
    scores = np.array([5, 4, 3, 8, 7, 3, 2, 4, 1])
    mask = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
    _, small_per, small = evaluate_fixed(frame, scores, mask, k=2)
    _, large_per, large = evaluate_fixed(frame, scores, np.ones(len(frame), dtype=bool), k=2)
    ledger = expansion_ledger(frame, small, large, mask, k=2).set_index("user_id")
    assert ledger.loc[1, "new_true_hits"] == ledger.loc[1, "lost_old_true_hits"] == 1
    assert ledger.loc[1, "delta_hits"] == 0
    assert ledger.loc[2, "new_true_hits"] == 1 and ledger.loc[2, "lost_old_true_hits"] == 0
    assert ledger.loc[2, "delta_recall"] == 0.5
    assert len(ledger) == 3 and np.isnan(ledger.loc[3, "delta_recall"])
    np.testing.assert_array_equal(ledger.delta_hits, large_per.hits - small_per.hits)
    np.testing.assert_allclose(ledger.new_true_recall - ledger.lost_old_true_recall, ledger.delta_recall, equal_nan=True)


def test_ledger_rejects_inconsistent_fixed_ranking(catalog):
    # All-pool expansion cannot replace a top item with an already-eligible item.
    small = catalog.loc[(catalog.user_id == 1) & (catalog.product_id == 10)]
    large = catalog.loc[(catalog.user_id == 1) & (catalog.product_id == 20)]
    with pytest.raises(ValueError, match="fixed-score"):
        expansion_ledger(catalog, small, large, np.ones(len(catalog), dtype=bool), k=1)


def test_randomized_metrics_match_independent_set_reference():
    rng = np.random.default_rng(913)
    records = []
    for user in range(30):
        for product in range(int(rng.integers(1, 20))):
            records.append((user, product, int(rng.integers(0, 2)), float(rng.integers(0, 6)), bool(rng.integers(0, 2))))
    frame = pd.DataFrame(records, columns=["user_id", "product_id", "label", "score", "eligible"])
    frame = frame.sample(frac=1, random_state=2)
    _, actual, _ = evaluate_fixed(frame, frame.score.to_numpy(), frame.eligible.to_numpy(), k=4)
    actual = actual.set_index("user_id")
    for user, group in frame.groupby("user_id"):
        true = {int(row.product_id) for row in group.itertuples() if row.label}
        allowed = {int(row.product_id) for row in group.itertuples() if row.eligible}
        ordered = sorted((row for row in group.itertuples() if row.eligible), key=lambda row: (-row.score, row.product_id))[:4]
        hits = sum(row.product_id in true for row in ordered)
        assert actual.loc[user, "repeat_targets"] == len(true)
        assert actual.loc[user, "candidates"] == len(allowed)
        assert actual.loc[user, "hits"] == hits
        assert actual.loc[user, "precision"] == (hits / len(ordered) if ordered else 0)
        assert actual.loc[user, "fixed_budget_precision"] == hits / 4
        if true:
            assert actual.loc[user, "recall"] == hits / len(true)
            assert actual.loc[user, "coverage"] == len(true & allowed) / len(true)
            assert actual.loc[user, "oracle_recall"] == min(4, len(true & allowed)) / len(true)
            dcg = sum(int(row.product_id in true) / np.log2(rank + 1) for rank, row in enumerate(ordered, 1))
            ideal = sum(1 / np.log2(rank + 1) for rank in range(1, min(4, len(true)) + 1))
            assert actual.loc[user, "ndcg"] == pytest.approx(dcg / ideal)
        else:
            assert np.isnan(actual.loc[user, "recall"]) and np.isnan(actual.loc[user, "ndcg"])
