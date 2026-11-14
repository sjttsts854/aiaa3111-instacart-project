import numpy as np
import pandas as pd
import pytest

from study4_followup.users import change_summary, top_overlap, candidate_overlap, difference_attributes


def test_changes_use_population_denominator_and_keep_empty_count():
    result = change_summary([.5, -.25, 0, np.nan], n_boot=0)
    assert result['users'] == 4
    assert result['nonempty_users'] == 3
    assert result['improved_fraction'] == 1/3
    assert result['harmed_fraction'] == 1/3
    assert result['unchanged_fraction'] == 1/3
    assert result['mean_delta'] == pytest.approx(1/12)
    assert result['gain_contribution'] == pytest.approx(1/6)
    assert result['loss_contribution'] == pytest.approx(1/12)
    assert result['changed_mean_delta'] == .125
    assert result['gain_contribution'] - result['loss_contribution'] == pytest.approx(result['mean_delta'])
    empty = change_summary([np.nan], n_boot=20)
    assert empty['nonempty_users'] == 0 and empty['mean_delta'] is None


def test_tolerance_only_changes_numerical_zeros_and_bootstrap_repeats():
    delta = [1e-14, -.1, .2, .3]
    a = change_summary(delta, n_boot=100, seed=7)
    b = change_summary(delta, n_boot=100, seed=7)
    assert a == b
    assert a['unchanged_users'] == 1
    assert a['ci95_low'] <= a['mean_delta'] <= a['ci95_high']


def test_top_overlap_separates_order_from_set_and_preserves_no_lists():
    a = pd.DataFrame({'user_id': [1, 1, 2], 'product_id': [10, 11, 12], 'rank': [1, 2, 1]})
    b = pd.DataFrame({'user_id': [1, 1, 2], 'product_id': [11, 10, 13], 'rank': [1, 2, 1]})
    values = top_overlap(a, b, [1, 2, 3]).set_index('user_id')
    assert not values.loc[1, 'ordered_identical'] and values.loc[1, 'unordered_identical']
    assert values.loc[2, 'jaccard'] == 0
    assert values.loc[3, 'jaccard'] == 1 and values.loc[3, 'ordered_identical']
    with pytest.raises(ValueError, match='unknown'):
        top_overlap(a, b, [1])


def test_candidate_overlap_distinguishes_shared_recent_from_different_old():
    frame = pd.DataFrame({'user_id': [1, 1, 1, 2], 'product_id': [10, 11, 12, 20],
                          'orders_since_purchase': [0, 8, 5, 1], 'purchase_count': [1, 4, 1, 2],
                          'label': [1, 0, 1, 0]})
    dual = np.array([True, True, False, True])
    recent = np.array([True, False, True, True])
    values = candidate_overlap(frame, dual, recent).set_index('user_id')
    assert values.loc[1, 'jaccard'] == 1/3
    assert values.loc[1, 'old_jaccard'] == 0
    assert np.isnan(values.loc[2, 'old_jaccard'])
    assert values.all_recent_in_both.all()
    attrs = difference_attributes(frame, dual, recent).set_index('candidate_part')
    assert attrs.loc['dual_only', 'row_mean_age'] == 8
    assert attrs.loc['recency_only', 'row_mean_frequency'] == 1
    changed = frame.assign(label=1-frame.label)
    pd.testing.assert_frame_equal(candidate_overlap(changed, dual, recent), candidate_overlap(frame, dual, recent))
    with pytest.raises(ValueError, match='matched'):
        candidate_overlap(frame, dual, np.ones(4, dtype=bool))
