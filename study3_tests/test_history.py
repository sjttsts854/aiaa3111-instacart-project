import numpy as np
import pandas as pd
import pytest

from study3.history import analyze_history, make_details, standardize_spec


def synthetic_details(within_gain=0.):
    rows = []
    uid = 0
    for group, easy, hard, offset in [("3-5", 40, 10, 0.), ("6-15", 25, 25, within_gain/2),
                                      ("16+", 10, 40, within_gain)]:
        for candidates, count, recall in [(15, easy, .8+offset), (80, hard, .2+offset)]:
            for _ in range(count):
                uid += 1
                rows.append({"user_id": uid, "history_group": group, "candidates": candidates,
                             "repeat_targets": 3, "recall": recall})
    return pd.DataFrame(rows)


def test_same_cells_remove_composition_without_changing_within_cell_outcomes():
    result, groups, cells = standardize_spec(synthetic_details(), "original", 10, n_boot=50)
    assert result["raw_overall_long_minus_short"] == pytest.approx(-.36)
    assert result["standardized_long_minus_short"] == pytest.approx(0.)
    assert result["standardized_ci95_low"] == pytest.approx(0.)
    assert result["standardized_ci95_high"] == pytest.approx(0.)
    np.testing.assert_allclose(groups.standardized_recall, .5)
    assert result["retained_users"] == 150 and result["eligible_cells"] == 2
    assert all(cells.groupby("history_group").pooled_weight.sum().eq(1.))


def test_within_cell_advantage_survives_standardization_and_bootstrap():
    result, _, _ = standardize_spec(synthetic_details(.1), "original", 10, n_boot=50)
    assert result["standardized_long_minus_short"] == pytest.approx(.1)
    assert result["standardized_ci95_low"] == pytest.approx(.1)
    assert result["standardized_ci95_high"] == pytest.approx(.1)
    assert result["point_reversal_within_common_support"]
    assert result["point_reversal_from_overall"]


def test_common_support_requires_minimum_for_every_group_and_handles_empty_support():
    result, groups, cells = standardize_spec(synthetic_details(), "original", 20, n_boot=20)
    assert result["eligible_cells"] == 0 and result["retained_users"] == 0
    assert result["standardized_long_minus_short"] is None
    assert groups.retained_users.sum() == 0 and cells.empty


def test_sample_restriction_and_reweighting_shifts_add_to_total_shift():
    df = synthetic_details(.1)
    extra = pd.DataFrame([{"user_id": 1000+i, "history_group": "16+", "candidates": 300,
                           "repeat_targets": 3, "recall": 0.} for i in range(20)])
    result, groups, _ = standardize_spec(pd.concat([df, extra]), "original", 10, n_boot=30)
    total_shift = result["standardized_long_minus_short"]-result["raw_overall_long_minus_short"]
    assert result["sample_restriction_shift"]+result["reweighting_shift_within_common_support"] == pytest.approx(total_shift)
    assert groups.loc[groups.history_group.eq("16+"), "retained_fraction"].iloc[0] == pytest.approx(50/70)
    assert result["retained_users"] == 150


def test_make_details_checks_target_semantics_and_candidate_population():
    candidate = pd.DataFrame({"user_id": [1, 1, 2], "product_id": [2, 3, 9], "history_orders": [3, 3, 20]})
    per = pd.DataFrame({"user_id": [1, 2], "recall": [1., np.nan], "repeat_targets": [1, 0]})
    result = make_details(per, candidate)
    assert result.candidates.tolist() == [2, 1]
    assert result.history_group.tolist() == ["3-5", "16+"]
    with pytest.raises(ValueError, match="Empty repeat"):
        make_details(per.fillna(0), candidate)
    with pytest.raises(ValueError, match="user sets"):
        make_details(per.iloc[:1], candidate)
    with pytest.raises(ValueError, match="Duplicate historical"):
        make_details(per, pd.concat([candidate, candidate.iloc[:1]]))


def test_analyze_writes_all_six_without_claiming_stability_when_support_missing(tmp_path):
    source = synthetic_details(.1)
    source["history_orders"] = source.history_group.map({"3-5": 4, "6-15": 8, "16+": 20})
    candidates = source.loc[source.index.repeat(source.candidates), ["user_id", "history_orders"]]
    out = analyze_history(source[["user_id", "recall", "repeat_targets"]], candidates, tmp_path, n_boot=20)
    assert out["specifications"] == 6
    assert not out["all_six_evaluable"] and not out["same_nonzero_direction_in_all_six"]
    assert len(pd.read_csv(tmp_path/"sensitivity.csv")) == 6
    assert (tmp_path/"method.txt").exists()
