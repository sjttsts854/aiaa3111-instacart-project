import numpy as np
import pandas as pd
import pytest

from study4_followup.audit import history_masks, ranked_rows, per_at_k, sign_summary, overlap_rows, pruning_ledger, verify_hashes


def frame():
    return pd.DataFrame({'user_id':[1,1,1,1,2,2], 'product_id':[1,2,3,4,1,2],
                         'label':[1,0,1,1,0,0], 'purchase_count':[1,10,2,3,1,1],
                         'orders_since_purchase':[0,8,6,9,1,0]})


def test_pruning_keeps_full_target_denominator_and_empty_target_users():
    data=frame();scores=np.array([.9,.8,.7,.6,.8,.9]);chosen=np.array([1,0,1,0,0,0],dtype=bool)
    full=ranked_rows(data,scores,np.ones(len(data),dtype=bool));small=ranked_rows(data,scores,chosen)
    values=per_at_k(data,small,2)
    assert values.loc[1,'repeat_targets']==3 and values.loc[1,'recall']==2/3
    assert values.loc[2,'recommended']==0 and np.isnan(values.loc[2,'recall'])
    ledger=pruning_ledger(data,full,small,2)
    assert ledger.loc[1,'promoted_true_hits']==1 and ledger.loc[1,'removed_true_hits']==0
    assert ledger.loc[1,'delta_recall']==1/3
    # A retained high-ranked item cannot disappear under the same scores.
    corrupt=small.copy();corrupt.loc[corrupt.product_id.eq(1),'rank']=3
    with pytest.raises(AssertionError,match='Retained'):
        pruning_ledger(data,full,corrupt,2)


def test_history_only_masks_match_budgets_even_with_changed_labels():
    data=frame();masks=history_masks(data,{'kind':'dual','window':5,'m':1})
    np.testing.assert_array_equal(masks['dual'],[1,1,0,0,1,1])
    np.testing.assert_array_equal(masks['recency_budget'],[1,0,1,0,1,1])
    data.label=1-data.label
    changed=history_masks(data,{'kind':'dual','window':5,'m':1})
    for name in masks:np.testing.assert_array_equal(masks[name],changed[name])


def test_zero_net_change_does_not_hide_individual_gain_and_loss():
    result=sign_summary([.2,-.2,0,np.nan])
    assert result['nonempty_users']==3
    assert result['improved_users']==result['worse_users']==result['unchanged_users']==1
    assert result['mean_delta']==0 and result['mean_absolute_delta']>0
    assert result['gain_contribution']==-result['loss_contribution']
    a=pd.DataFrame({'user_id':[1,1], 'product_id':[1,2]})
    b=pd.DataFrame({'user_id':[1,1], 'product_id':[2,1]})
    overlap=overlap_rows(a,b,[1,2])
    assert overlap.loc[1,'jaccard']==1 and overlap.loc[1,'unordered_equal'] and not overlap.loc[1,'ordered_equal']
    assert overlap.loc[2,'ordered_equal'] and overlap.loc[2,'jaccard']==1


def test_changed_input_is_rejected(tmp_path):
    path=tmp_path/'artifact.txt';path.write_text('changed')
    with pytest.raises(AssertionError,match='protected input'):
        verify_hashes({str(path):'0'*64})
