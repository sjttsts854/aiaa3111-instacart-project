import numpy as np
import pandas as pd
import pytest

from basketlab.metrics import evaluate
from study3.quota import quota_scores


def top(frame, scores, k):
    return (frame[["user_id", "product_id", "co_narrow_covered"]]
            .assign(score=scores)
            .sort_values(["user_id", "score", "product_id"], ascending=[True, False, True])
            .groupby("user_id", sort=False).head(k))


def test_quota_stops_cross_group_displacement_but_allows_within_group_improvement():
    # Base chooses covered product 1 and uncovered product 3. Rule model pushes
    # both covered products to the top, losing correct uncovered product 3.
    f = pd.DataFrame({"user_id": [1]*4, "product_id": [1, 2, 3, 4],
                      "co_narrow_covered": [1, 1, 0, 0], "label": [0, 1, 1, 0]})
    base = np.array([.9, .7, .8, .6])
    rule = np.array([.95, .99, .5, .4])
    score = quota_scores(f, base, rule, k=2)
    assert top(f, base, 2).product_id.tolist() == [1, 3]
    assert top(f, rule, 2).product_id.tolist() == [2, 1]
    assert top(f, score, 2).product_id.tolist() == [2, 3]
    assert evaluate(f, score, 2)[0]["recall_at_2"] == 1
    assert evaluate(f, rule, 2)[0]["recall_at_2"] == .5


def test_zero_weight_exactly_returns_base_scores_and_does_not_mutate_inputs():
    f = pd.DataFrame({"user_id": [2, 1, 1], "product_id": [3, 2, 1],
                      "co_narrow_covered": [1, 0, 1]}, index=[7, 7, 3])
    before = f.copy(deep=True)
    base = np.array([-.4, 9., 9.])
    result = quota_scores(f, base, [4., 1., 2.], k=1, weight=0)
    np.testing.assert_array_equal(result, base)
    assert result is not base
    pd.testing.assert_frame_equal(f, before)
    assert top(f, result, 1).product_id.tolist() == [1, 3]


@pytest.mark.parametrize("coverage", [0, 1])
def test_one_coverage_group_reduces_to_rule_ranking(coverage):
    f = pd.DataFrame({"user_id": [1]*4, "product_id": [4, 2, 1, 3],
                      "co_narrow_covered": [coverage]*4})
    rule = [.5, .8, .8, .1]
    result = quota_scores(f, [4, 3, 2, 1], rule, k=2)
    assert top(f, result, 4).product_id.tolist() == [1, 2, 4, 3]


def test_small_pools_and_equal_scores_have_stable_product_ties():
    f = pd.DataFrame({"user_id": [1, 1, 1, 2], "product_id": [8, 2, 4, 6],
                      "co_narrow_covered": [0, 1, 0, 1]})
    result = quota_scores(f, np.ones(4), np.ones(4), k=10)
    assert top(f, result, 10).product_id.tolist() == [2, 4, 8, 6]
    assert np.isfinite(result).all()


def test_target_labels_have_no_effect():
    f = pd.DataFrame({"user_id": [1]*4, "product_id": range(4),
                      "co_narrow_covered": [0, 1, 0, 1], "label": [1, 0, 1, 0]})
    a = quota_scores(f, [4, 3, 2, 1], [1, 4, 3, 2], k=2)
    f.label = 1-f.label
    b = quota_scores(f, [4, 3, 2, 1], [1, 4, 3, 2], k=2)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("weight", [.25, .5, 1.])
@pytest.mark.parametrize("k", [1, 5, 10, 20])
def test_randomized_count_invariance_and_exact_group_optimum(k, weight):
    rng = np.random.default_rng(20261114)
    sizes = rng.integers(1, 60, size=80)
    f = pd.DataFrame({"user_id": np.repeat(np.arange(len(sizes)), sizes)})
    f["product_id"] = f.groupby("user_id").cumcount()
    f["co_narrow_covered"] = rng.integers(0, 2, len(f))
    f = f.sample(frac=1, random_state=13).reset_index(drop=True)
    base = rng.integers(-3, 5, len(f)).astype(float)
    rule = rng.integers(-3, 5, len(f)).astype(float)
    result = quota_scores(f, base, rule, k=k, weight=weight)
    bt = top(f, base, k); qt = top(f, result, k)
    pd.testing.assert_series_equal(bt.groupby("user_id").co_narrow_covered.sum(),
                                   qt.groupby("user_id").co_narrow_covered.sum())
    np.testing.assert_array_equal(qt.groupby("user_id").size(), np.minimum(sizes, k))
    # Independently enumerate each user's two groups to verify both optimum and
    # product-id tie handling; this is separate from the vectorized implementation.
    mixed = (1-weight)*base+weight*rule
    for user, sub in f.groupby("user_id"):
        base_items = sorted(sub.index, key=lambda i: (-base[i], f.at[i, "product_id"]))[:k]
        expected = []
        for covered in [0, 1]:
            n = sum(f.at[i, "co_narrow_covered"] == covered for i in base_items)
            eligible = sub.index[sub.co_narrow_covered.eq(covered)]
            expected += sorted(eligible, key=lambda i: (-mixed[i], f.at[i, "product_id"]))[:n]
        expected.sort(key=lambda i: (-mixed[i], f.at[i, "product_id"]))
        assert qt.loc[qt.user_id.eq(user), "product_id"].tolist() == f.loc[expected, "product_id"].tolist()


@pytest.mark.parametrize("change", ["duplicate", "nan", "coverage", "scores", "shape", "k", "weight"])
def test_invalid_inputs_rejected(change):
    f = pd.DataFrame({"user_id": [1, 1], "product_id": [1, 2], "co_narrow_covered": [1, 0]})
    base = [.8, .7]; rule = [.5, .9]; kwargs = {}
    if change == "duplicate": f.product_id = [1, 1]
    if change == "nan": f.loc[0, "user_id"] = np.nan
    if change == "coverage": f.co_narrow_covered = [1, 2]
    if change == "scores": rule = [np.inf, 2]
    if change == "shape": base = [[.8, .7]]
    if change == "k": kwargs["k"] = 1.5
    if change == "weight": kwargs["weight"] = -1
    with pytest.raises(ValueError): quota_scores(f, base, rule, **kwargs)


def test_empty_input_returns_finite_empty_scores():
    f = pd.DataFrame(columns=["user_id", "product_id", "co_narrow_covered"])
    result = quota_scores(f, [], [], k=3)
    assert result.shape == (0,) and np.isfinite(result).all()
