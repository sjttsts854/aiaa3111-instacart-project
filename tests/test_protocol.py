import numpy as np
import pandas as pd
import pytest

from basketlab.data import split_users, build_candidates
from basketlab.metrics import evaluate, attach_labels, paired_bootstrap
from basketlab.enrichment import fit_rules, rule_features, fit_clusters
from basketlab.models import feature_columns


def fixture_history():
    orders = pd.DataFrame({"order_id":[1,2,3,4], "user_id":[10,10,20,20],
                           "eval_set":["prior"]*4,"order_number":[1,2,1,2],
                           "days_since_prior_order":[np.nan,7,np.nan,10]})
    history = pd.DataFrame({"order_id":[1,1,2,3,4],"product_id":[1,2,1,2,3],
                            "add_to_cart_order":[1,2,1,1,1],"reordered":[0,0,1,0,0]})
    products = pd.DataFrame({"product_id":[1,2,3,4],"product_name":["A","B","C","D"],
                            "aisle_id":[1,1,2,2],"department_id":[1,1,2,2]})
    return history, orders, products


def test_split_disjoint_reproducible():
    a = split_users(np.arange(100),100,42)
    pd.testing.assert_frame_equal(a,split_users(np.arange(100)[::-1],100,42))
    assert a.user_id.nunique()==100
    assert a.split.value_counts().to_dict()=={"train":70,"valid":15,"test":15}


def test_feature_values_and_candidates():
    h,o,p=fixture_history(); c,_,_=build_candidates(h,o,p,[10])
    row=c[(c.user_id==10)&(c.product_id==1)].iloc[0]
    assert row.purchase_count==2 and row.frequency==1
    assert row.in_last_basket==1 and row.orders_since_purchase==0
    assert np.isclose(row.decay_score,1.8)
    assert row.global_support==1
    assert set(c[c.user_id==10].product_id)=={1,2}
    assert 4 not in c.product_id.values
    assert c[(c.user_id==20)&(c.product_id==3)].unseen_in_training.item()==1


def test_validation_history_does_not_change_training_features():
    h,o,p=fixture_history();a,_,_=build_candidates(h,o,p,[10])
    extra=pd.DataFrame({"order_id":[4],"product_id":[4],"add_to_cart_order":[2],"reordered":[0]})
    b,_,_=build_candidates(pd.concat([h,extra]),o,p,[10])
    pd.testing.assert_frame_equal(a[a.user_id==10].reset_index(drop=True),b[b.user_id==10].reset_index(drop=True))


def test_future_orders_rejected():
    h,o,p=fixture_history();o.loc[0,"eval_set"]="train"
    with pytest.raises(ValueError,match="prior"):build_candidates(h,o,p,[10])


def test_target_labels_do_not_add_candidates():
    c=pd.DataFrame({"user_id":[1,1],"product_id":[1,2]})
    labels=pd.DataFrame({"user_id":[1,1],"product_id":[1,3]})
    out=attach_labels(c,labels)
    assert out.product_id.tolist()==[1,2] and out.label.tolist()==[1,0]


def test_metrics_hand_computed_and_empty_target():
    c=pd.DataFrame({"user_id":[1,1,1,2],"product_id":[1,2,3,1],"label":[1,0,1,0]})
    metrics,per,top=evaluate(c,np.array([.9,.8,.7,.1]),k=2)
    assert metrics["recall_at_2"]==.5
    assert metrics["precision_at_2"]==.25
    assert metrics["f1_at_2"]==.25
    assert metrics["empty_repeat_targets"]==1
    assert np.isclose(metrics["ndcg_at_2"],1/(1+1/np.log2(3)))
    assert np.isnan(per[per.user_id==2].recall.item())


def test_ties_deterministic_and_short_candidates():
    c=pd.DataFrame({"user_id":[1,1],"product_id":[2,1],"label":[0,1]})
    metrics,_,top=evaluate(c,np.array([1.,1.]))
    assert top.product_id.tolist()==[1,2]
    assert metrics["recall_at_10"]==1 and metrics["mean_recommendation_length"]==2
    with pytest.raises(ValueError):evaluate(c,np.array([np.nan,1.]))


def test_rule_fit_excludes_heldout_users():
    h=pd.DataFrame({"user_id":[1,1,1,1,2,2],"order_id":[1,1,2,2,3,3],"product_id":[1,2,1,2,8,9]})
    rules,info=fit_rules(h,[1],top_n=20,min_support=.1,min_count=1)
    assert info["training_baskets"]==2 and info["top_products"]==2
    assert not rules.antecedent.isin([8,9]).any()


def test_rule_features_use_observed_last_basket():
    h=pd.DataFrame({"user_id":[1,1],"product_id":[1,2],"age":[1,0]})
    rules=pd.DataFrame({"antecedent":[1,2],"consequent":[3,4],"confidence":[.4,.7],"lift":[2.,3.]})
    f=rule_features(h,rules)
    assert f.product_id.tolist()==[4] and f.rule_confidence.item()==.7


def test_model_features_exclude_labels_and_identifiers():
    f=pd.DataFrame(columns=["label","user_id","product_id","split","cluster_distance_0"])
    cols=feature_columns(f,{"clusters":True,"rules":True})
    assert not set(cols)&{"label","user_id","product_id","split","order_dow","order_hour_of_day"}


def test_paired_bootstrap_zero_difference():
    a=pd.DataFrame({"user_id":[1,2,3],"recall":[.5,1.,np.nan]})
    out=paired_bootstrap(a,a,n_boot=100)
    assert out["mean_delta"]==0 and out["ci95"]==[0.,0.] and out["users"]==2


def test_actual_association_rule_values():
    h=pd.DataFrame({"user_id":[1]*6,"order_id":[1,1,2,2,3,4],"product_id":[1,2,1,2,3,3]})
    rules,_=fit_rules(h,[1],top_n=3,min_support=.1,min_count=1)
    rule=rules[(rules.antecedent==1)&(rules.consequent==2)].iloc[0]
    assert rule.support==.5 and rule.confidence==1. and rule.lift==2.


def test_cluster_fit_is_invariant_to_heldout_history():
    users=pd.DataFrame({"user_id":range(9),"split":["train"]*8+["valid"],
        "dept_share_1":[.1,.11,.09,.12,.8,.81,.79,.82,.4],
        "basket_mean":[2,2,3,2,10,11,10,9,5],"interval_mean":[5]*9,
        "historical_repeat_rate":[.4]*9,"history_orders":[10]*9})
    a,_,pa,_=fit_clusters(users,n_clusters=2)
    users.loc[8,"basket_mean"]=10000
    b,_,pb,_=fit_clusters(users,n_clusters=2)
    np.testing.assert_array_equal(pa["scaler"].mean_,pb["scaler"].mean_)
    pd.testing.assert_frame_equal(a.iloc[:8],b.iloc[:8])


def test_macro_recall_weights_users_equally():
    c=pd.DataFrame({"user_id":[1,1,1,1,2],"product_id":[1,2,3,4,1],"label":[1,1,1,1,1]})
    summary,_,_=evaluate(c,np.array([.9,.8,.7,.6,.5]),k=1)
    assert summary["recall_at_1"]==(.25+1)/2
