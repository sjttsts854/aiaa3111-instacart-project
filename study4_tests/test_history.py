import json

import numpy as np
import pandas as pd
import pytest

from study4.history import BINNING, analyze_observed, make_details, standardize_spec


def synthetic_details(within_gain=0., scale=1):
    rows = []
    uid = 0
    for group, easy, hard, offset in [("3-5", 40, 10, 0.), ("6-15", 25, 25, within_gain / 2),
                                      ("16+", 10, 40, within_gain)]:
        for candidates, basket, count, recall in [(15, 4., easy, .8 + offset), (80, 12., hard, .2 + offset)]:
            for _ in range(count * scale):
                uid += 1
                rows.append({"user_id": uid, "history_group": group, "candidates": candidates,
                             "basket_mean": basket, "repeat_targets": 3, "recall": recall})
    return pd.DataFrame(rows)


def as_inputs(details):
    source = details.copy()
    source["history_orders"] = source.history_group.map({"3-5": 4, "6-15": 8, "16+": 20})
    frame = source.loc[source.index.repeat(source.candidates), ["user_id", "history_orders", "basket_mean"]].copy()
    frame["product_id"] = frame.groupby("user_id").cumcount() + 1
    return source[["user_id", "recall", "repeat_targets"]], frame


@pytest.mark.parametrize("binning", BINNING)
def test_equal_within_cell_outcomes_remove_simpson_composition_effect(binning):
    result, groups, cells = standardize_spec(synthetic_details(), binning, 10, n_boot=50)
    assert result["raw_overall_long_minus_short"] == pytest.approx(-.36)
    assert result["standardized_long_minus_short"] == pytest.approx(0.)
    assert result["standardized_ci95_low"] == pytest.approx(0.)
    assert result["standardized_ci95_high"] == pytest.approx(0.)
    np.testing.assert_allclose(groups.standardized_recall, .5)
    assert result["retained_users"] == 150 and result["eligible_cells"] == 2
    assert cells.groupby("history_group").pooled_weight.sum().eq(1.).all()


@pytest.mark.parametrize("binning", BINNING)
def test_within_cell_advantage_survives_standardization_and_bootstrap(binning):
    result, _, _ = standardize_spec(synthetic_details(.1), binning, 10, n_boot=50)
    assert result["raw_overall_long_minus_short"] == pytest.approx(-.26)
    assert result["standardized_long_minus_short"] == pytest.approx(.1)
    assert result["standardized_ci95_low"] == pytest.approx(.1)
    assert result["standardized_ci95_high"] == pytest.approx(.1)
    assert result["point_reversal_within_common_support"]
    assert result["point_reversal_from_overall"]


@pytest.mark.parametrize("binning", BINNING)
def test_positive_future_target_counts_cannot_change_support_weights_or_contrasts(binning):
    per, frame = as_inputs(synthetic_details(.1))
    before = standardize_spec(make_details(per, frame), binning, 10, n_boot=37)
    # Large changes cross every old future-target bin while preserving which
    # users have defined Recall. Neither the outcome nor any historical field changes.
    per["repeat_targets"] = np.where(per.user_id.mod(2).eq(0), 1, 15)
    frame["future_basket_size"] = 999  # An irrelevant terminal field is ignored.
    after = standardize_spec(make_details(per, frame), binning, 10, n_boot=37)
    assert before[0] == after[0]
    pd.testing.assert_frame_equal(before[1], after[1])
    pd.testing.assert_frame_equal(before[2], after[2])
    assert not any("target" in column for column in after[2].columns)


def test_historical_basket_strata_add_real_conditioning_beyond_candidates():
    details = synthetic_details(.1)
    details["candidates"] = 80
    only, _, only_cells = standardize_spec(details, "candidate_only", 10, n_boot=30)
    crossed, _, crossed_cells = standardize_spec(details, "candidate_historical_basket", 10, n_boot=30)
    assert only["standardized_long_minus_short"] == pytest.approx(-.26)
    assert crossed["standardized_long_minus_short"] == pytest.approx(.1)
    assert only["eligible_cells"] == 1 and crossed["eligible_cells"] == 2
    assert only_cells.historical_basket_bin.eq("not_conditioned").all()
    assert crossed_cells.historical_basket_lower.notna().all()


def test_bootstrap_resamples_users_and_is_reproducible_for_fixed_seed():
    details = synthetic_details(.1)
    details["recall"] += np.random.default_rng(1).uniform(-.08, .08, len(details))
    first = standardize_spec(details, "candidate_only", 10, n_boot=400, seed=9)[0]
    repeated = standardize_spec(details, "candidate_only", 10, n_boot=400, seed=9)[0]
    assert first == repeated
    assert first["standardized_ci95_low"] < first["standardized_long_minus_short"] < first["standardized_ci95_high"]
    assert first["raw_common_support_ci95_low"] < first["raw_common_support_long_minus_short"] < first["raw_common_support_ci95_high"]


@pytest.mark.parametrize("binning", BINNING)
def test_common_support_requires_minimum_for_every_group_and_handles_empty_support(binning):
    result, groups, cells = standardize_spec(synthetic_details(), binning, 20, n_boot=20)
    assert result["eligible_cells"] == 0 and result["retained_users"] == 0
    assert result["standardized_long_minus_short"] is None
    assert result["raw_common_support_long_minus_short"] is None
    assert groups.retained_users.sum() == 0 and cells.empty
    assert "pooled_weight" in cells.columns


def test_sample_restriction_and_reweighting_are_separate_additive_shifts():
    df = synthetic_details(.1)
    extra = pd.DataFrame([{"user_id": 1000 + i, "history_group": "16+", "candidates": 300,
                           "basket_mean": 12., "repeat_targets": 3, "recall": 0.} for i in range(20)])
    result, groups, _ = standardize_spec(pd.concat([df, extra]), "candidate_only", 10, n_boot=30)
    total_shift = result["standardized_long_minus_short"] - result["raw_overall_long_minus_short"]
    assert result["sample_restriction_shift"] + result["reweighting_shift_within_common_support"] == pytest.approx(total_shift)
    assert groups.loc[groups.history_group.eq("16+"), "retained_fraction"].iloc[0] == pytest.approx(50 / 70)
    assert result["retained_users"] == 150


def test_make_details_checks_historical_inputs_and_target_semantics():
    frame = pd.DataFrame({"user_id": [1, 1, 2], "product_id": [2, 3, 9],
                          "history_orders": [3, 3, 20], "basket_mean": [1.5, 1.5, 1.]})
    per = pd.DataFrame({"user_id": [1, 2], "recall": [1., np.nan], "repeat_targets": [1, 0]})
    result = make_details(per, frame)
    assert result.candidates.tolist() == [2, 1]
    assert result.history_group.tolist() == ["3-5", "16+"]
    with pytest.raises(ValueError, match="Empty repeat"):
        make_details(per.fillna(0), frame)
    with pytest.raises(ValueError, match="user sets"):
        make_details(per.iloc[:1], frame)
    with pytest.raises(ValueError, match="Duplicate historical"):
        make_details(per, pd.concat([frame, frame.iloc[:1]]))
    inconsistent = frame.copy()
    inconsistent.loc[0, "basket_mean"] = 2.
    with pytest.raises(ValueError, match="Inconsistent historical basket_mean"):
        make_details(per, inconsistent)
    with pytest.raises(ValueError, match="finite and positive"):
        make_details(per, frame.assign(basket_mean=np.inf))
    with pytest.raises(ValueError, match="Invalid repeat-target"):
        make_details(per.assign(repeat_targets=4), frame)


def test_all_empty_targets_are_reported_without_crash(tmp_path):
    per, frame = as_inputs(synthetic_details())
    per["repeat_targets"] = 0
    per["recall"] = np.nan
    result = analyze_observed(per, frame, tmp_path, n_boot=20)
    assert result["nonempty_users"] == 0 and result["empty_target_users"] == 150
    assert result["evaluable_specifications"] == 0
    assert result["standardized_delta_min"] is None
    assert pd.read_csv(tmp_path / "cells.csv").empty
    raw = pd.read_csv(tmp_path / "raw_overall.csv")
    assert raw.empty_target_users.sum() == 150
    assert raw.recall.isna().all()


def test_analyze_writes_all_four_and_does_not_claim_stability_when_support_missing(tmp_path):
    per, frame = as_inputs(synthetic_details(.1))
    result = analyze_observed(per, frame, tmp_path, n_boot=20)
    assert result["specifications"] == 4 and result["evaluable_specifications"] == 2
    assert not result["all_four_evaluable"] and not result["same_nonzero_direction_in_all_four"]
    assert result["future_target_size_used_for_strata"] is False
    assert len(pd.read_csv(tmp_path / "sensitivity.csv")) == 4
    assert len(pd.read_csv(tmp_path / "groups.csv")) == 12
    assert "nonempty future repeat target" in (tmp_path / "method.txt").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "summary.json").read_text()) == result
    assert json.loads((tmp_path / "specifications.json").read_text())["binning"] == BINNING


def test_analyze_can_report_all_four_evaluable(tmp_path):
    per, frame = as_inputs(synthetic_details(.1, scale=2))
    result = analyze_observed(per, frame, tmp_path, n_boot=20)
    assert result["all_four_evaluable"]
    assert result["all_four_standardized_positive"]
    assert result["all_four_reversed_within_common_support"]
    assert result["standardized_delta_min"] == pytest.approx(.1)
    assert result["standardized_delta_max"] == pytest.approx(.1)


@pytest.mark.parametrize("n_boot", [0, -1, 1.5, True])
def test_bootstrap_count_validation(n_boot):
    with pytest.raises(ValueError, match="n_boot must be a positive integer"):
        standardize_spec(synthetic_details(), "candidate_only", 10, n_boot=n_boot)
