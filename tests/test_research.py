import numpy as np
import pandas as pd

from research.data import fresh_split
from research.patterns import pair_rules,score_rules
from research.models import columns, RECENCY, CADENCE, INTERACTION
from basketlab.enrichment import fit_rules


def test_fresh_confirmation_excludes_every_initial_user():
    cfg={"split_seed":9,"train_users":20,"validation_users":5,"confirmation_users":8}
    a=fresh_split(np.arange(100),np.arange(40),cfg)
    assert len(a)==33 and a.user_id.nunique()==33
    assert not a.user_id.isin(range(40)).any()
    assert a.split.value_counts().to_dict()=={"train":20,"confirm":8,"valid":5}


def fixture_pairs():
    return pd.DataFrame({"user_id":[1,1,1,1,1,2,2],"order_id":[1,1,2,2,3,4,5],
                         "order_number":[1,1,2,2,3,1,2],"product_id":[1,2,1,2,3,8,9],
                         "age":[2,2,1,1,0,1,0]})


def test_exact_pairs_match_fpgrowth():
    h=fixture_pairs()
    a,_=fit_rules(h,[1],top_n=20,min_support=.01,min_count=1)
    b,_=pair_rules(h,[1],min_support=.01,min_count=1,top_n=20)
    keys=["antecedent","consequent"]
    a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys],b[keys],check_dtype=False)
    np.testing.assert_allclose(a[["support","confidence","lift"]],b[["support","confidence","lift"]])


def test_sequential_pairs_never_cross_users():
    h=fixture_pairs()
    r,m=pair_rules(h,[1,2],mode="sequential",min_support=.01,min_count=1,min_lift=0)
    assert m["contexts"]==3
    assert not ((r.antecedent==3)&(r.consequent==8)).any()
    assert ((r.antecedent==8)&(r.consequent==9)).any()
    assert not (r.antecedent==r.consequent).any()


def test_pair_mining_training_only():
    h=fixture_pairs()
    r,m=pair_rules(h,[1],mode="sequential",min_support=.01,min_count=1,min_lift=0)
    assert m["contexts"]==2
    assert not r.antecedent.isin([8,9]).any()


def test_scoring_preserves_candidate_pool_and_last_context():
    h=fixture_pairs();c=pd.DataFrame({"user_id":[1,1,2],"product_id":[2,99,9]})
    r=pd.DataFrame({"antecedent":[3,1,9],"consequent":[2,99,9],"confidence":[.5,1.,.7],"lift":[2.,5.,3.]})
    f=score_rules(h,c,r,"test")
    pd.testing.assert_frame_equal(f[["user_id","product_id"]],c)
    np.testing.assert_allclose(f.test_confidence_sum,[.5,0,.7])
    assert f.test_covered.tolist()==[1,0,1]


def test_no_temporal_ablation_removes_recency_proxies():
    cols=columns("no_temporal",pd.DataFrame())
    assert not set(cols)&set(RECENCY+CADENCE+INTERACTION+["first_purchase","last_purchase"])
    assert not set(columns("no_cadence",pd.DataFrame()))&set(CADENCE+INTERACTION)
