import numpy as np
import pandas as pd

from study3.audit import independent_metrics


def test_raw_set_metrics_ignore_stored_labels_and_handle_empty_targets():
    # labels are deliberately incorrect: the auditor must use raw repeat sets.
    top = pd.DataFrame({"user_id": [1, 1, 2], "product_id": [10, 11, 20], "label": [1, 0, 1]})
    per = independent_metrics(top, {1: {11, 12}, 2: set()})
    assert per.loc[1, "hits"] == 1
    assert per.loc[1, "recall"] == .5
    assert per.loc[1, "precision"] == .5
    assert per.loc[1, "f1"] == .5
    np.testing.assert_allclose(per.loc[1, "ndcg"], (1/np.log2(3))/(1+1/np.log2(3)))
    assert np.isnan(per.loc[2, "recall"]) and np.isnan(per.loc[2, "ndcg"])
    assert per.loc[2, "precision"] == 0 and per.loc[2, "f1"] == 0


def test_small_pool_metric_uses_actual_recommendation_length():
    top = pd.DataFrame({"user_id": [1], "product_id": [10]})
    per = independent_metrics(top, {1: {10}})
    for field in ["hits", "recommended", "recall", "precision", "f1", "ndcg"]:
        assert per.loc[1, field] == 1
