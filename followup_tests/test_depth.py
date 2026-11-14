import numpy as np
import pandas as pd
import pytest

from study4.memory import evaluate_fixed
from study4_followup.depth import pruning_ledger, target_rank_counts


def fixture():
    # User 1: a deleted false top item admits a true item below K.
    # User 2: both true targets are deleted, including one originally below K.
    # User 3: no repeat targets, and its sole candidate is removed.
    frame = pd.DataFrame({'user_id': [1,1,1,1,2,2,2,3], 'product_id': [10,11,12,13,20,21,22,30],
                          'label': [0,1,1,0,1,0,1,0],
                          'orders_since_purchase': [8,0,3,6,7,0,11,9],
                          'purchase_count': [1,3,1,4,3,1,1,1]})
    scores = np.array([.9,.8,.7,.6,.9,.8,.7,.9])
    mask = np.array([False,True,True,True,False,True,False,False])
    return frame, scores, mask


def test_pruning_ledger_conserves_all_users_full_targets_and_hit_change():
    frame, scores, mask = fixture()
    _, full, ft = evaluate_fixed(frame, scores, np.ones(len(frame), dtype=bool), k=2)
    _, pruned, pt = evaluate_fixed(frame, scores, mask, k=2)
    ledger = pruning_ledger(frame, ft, pt, mask, k=2).set_index('user_id')
    assert ledger.index.tolist() == [1,2,3]
    assert ledger.repeat_targets.tolist() == [2,2,0]
    assert ledger.promoted_true_hits.tolist() == [1,0,0]
    assert ledger.deleted_true_hits.tolist() == [0,1,0]
    np.testing.assert_allclose(ledger.delta_recall, [.5,-.5,np.nan], equal_nan=True)
    np.testing.assert_allclose(ledger.delta_recall,
                               pruned.set_index('user_id').recall-full.set_index('user_id').recall,
                               equal_nan=True)
    # The target outside both top lists remains in the denominator for user 2.
    assert pruned.set_index('user_id').loc[2, 'repeat_targets'] == 2
    assert ledger.loc[3, 'retained_hits'] == 0


def test_pruning_ledger_rejects_changed_order_and_forged_labels():
    frame, scores, mask = fixture()
    _, _, ft = evaluate_fixed(frame, scores, np.ones(len(frame), dtype=bool), k=2)
    _, _, pt = evaluate_fixed(frame, scores, mask, k=2)
    changed = pt.loc[~pt.product_id.eq(11)].copy()
    with pytest.raises(ValueError, match='scores or tie rules changed'):
        pruning_ledger(frame, ft, changed, mask, 2)
    changed = pt.copy(); changed.loc[changed.product_id.eq(12), 'label'] = 0
    with pytest.raises(ValueError, match='labels differ'):
        pruning_ledger(frame, ft, changed, mask, 2)


def test_target_rank_partition_keeps_removed_targets_and_band_boundaries():
    frame = pd.DataFrame({'user_id': [1]*52, 'product_id': np.arange(1,53),
                          'label': [int(i in [10,11,20,21,50,51]) for i in range(1,53)],
                          'orders_since_purchase': [7]*52, 'purchase_count': [1]*52})
    scores = -frame.product_id.to_numpy(dtype=float)
    mask = frame.product_id.le(20).to_numpy()
    counts = target_rank_counts(frame, scores, mask)
    assert counts.targets.sum() == 6
    assert counts.groupby('eligibility').targets.sum().to_dict() == {'excluded':3,'retained':3}
    assert counts.groupby('rank_band').targets.sum().to_dict() == {'1-10':1,'11-20':2,'21-50':2,'>50':1}
    assert len(counts) == 24
    assert counts.loc[counts.product_group.ne('dormant_low_frequency'), 'targets'].sum() == 0


def test_target_rank_ties_use_product_id_not_input_order():
    frame = pd.DataFrame({'user_id':[1]*12, 'product_id':np.arange(12,0,-1),
                          'label':[1]+[0]*11, 'orders_since_purchase':[0]*12, 'purchase_count':[1]*12})
    counts = target_rank_counts(frame, np.ones(12), np.ones(12,dtype=bool))
    row = counts[counts.targets.eq(1)].iloc[0]
    assert row.rank_band == '11-20' and row.eligibility == 'retained' and row.product_group == 'recent'
