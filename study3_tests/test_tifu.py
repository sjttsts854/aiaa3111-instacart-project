import numpy as np
import pandas as pd
import pytest

from study3.tifu import order_weights,build_vectors,fit,find_neighbors,components,combine


SETTINGS={"group_count":2,"within_decay":.5,"group_decay":.25,"implementation":"author_code"}


def history(baskets):
    return pd.DataFrame([{"user_id":u,"order_number":i,"product_id":p,"eval_set":"prior"}
                         for u,orders in baskets.items() for i,items in enumerate(orders,1) for p in items])


def test_author_weights_hand_calculation_unequal_recent_groups():
    # Five orders, two groups with sizes 2 and 3. Global rb exponent 4..0.
    expected=np.array([(.5**4)*.25/4,(.5**3)*.25/4,.5**2/6,.5/6,1/6])
    np.testing.assert_allclose(order_weights(5,SETTINGS),expected)


def test_short_histories_keep_author_nominal_group_exponent():
    cfg={"group_count":3,"within_decay":.5,"group_decay":.25}
    np.testing.assert_allclose(order_weights(2,cfg),[.5*.25**2/2,.25/2])


def test_sparse_vectors_match_independent_loop_reference():
    rng=np.random.default_rng(99)
    baskets={u:[rng.choice(25,size=3,replace=False).tolist() for _ in range(u+1)] for u in range(1,13)}
    h=history(baskets)
    for m in [3,7]:
        cfg={"group_count":m,"within_decay":.9,"group_decay":.7}
        users,matrix=build_vectors(h,cfg,n_items=25)
        expected=[]
        for u in users:
            orders=baskets[u];n=len(orders);g=min(n,m)
            short=n//g;extra=n%g
            sizes=[short]*(g-extra)+[short+1]*extra
            result=np.zeros(25);position=0
            for group,size in enumerate(sizes):
                block=np.zeros(25)
                for _ in range(size):
                    for p in orders[position]:block[p]+=.9**(n-position-1)
                    position+=1
                result+=(block/size)*(.7**(m-group-1))/g
            expected.append(result)
        np.testing.assert_allclose(matrix.toarray(),expected,rtol=1e-12,atol=1e-14)


def test_bank_uses_training_history_only_and_excludes_self():
    h=history({1:[[1],[1]],2:[[2],[2]],3:[[1],[1]],99:[[9],[9]]})
    bank=fit(h,[1,2],SETTINGS,n_items=10)
    assert bank["train_user_ids"].tolist()==[1,2]
    assert bank["matrix"][:,9].nnz==0
    users,q=build_vectors(h[h.user_id.isin([1,3])],SETTINGS,n_items=10)
    idx,dist=find_neighbors(q,users,bank,k=1,batch_size=1)
    assert bank["train_user_ids"][idx].tolist()==[[2],[1]]
    assert dist[1,0]==pytest.approx(0)


def test_euclidean_neighbor_mean_and_candidate_row_alignment():
    h=history({1:[[1],[1]],2:[[2],[2]],3:[[1,2],[1]],4:[[1,2],[2]]})
    bank=fit(h,[1,2],SETTINGS,n_items=5)
    qh=h[h.user_id.isin([3,4])]
    c=pd.DataFrame({"user_id":[4,3,4,3],"product_id":[2,2,1,1]})
    a=components(qh,c,bank,neighbor_counts=[1,2],batch_size=1)
    b=components(qh,c,bank,neighbor_counts=[1,2],batch_size=20)
    users,q=build_vectors(qh,SETTINGS,n_items=5)
    dense=bank["matrix"].toarray();query=q.toarray()
    for i,row in c.iterrows():
        u=np.flatnonzero(users==row.user_id)[0]
        assert a["personal"][i]==pytest.approx(query[u,row.product_id])
        distances=((query[u]-dense)**2).sum(axis=1)
        nearest=np.lexsort((bank["train_user_ids"],distances))
        for k in [1,2]:
            assert a["neighbors"][k][i]==pytest.approx(dense[nearest[:k],row.product_id].mean())
            np.testing.assert_array_equal(a["neighbors"][k],b["neighbors"][k])
    np.testing.assert_allclose(combine(a,2,.7),.7*a["personal"]+.3*a["neighbors"][2])
    np.testing.assert_array_equal(combine(a,1,1),a["personal"])


def test_boundary_distance_ties_choose_lower_user_id_and_are_batch_invariant():
    h=history({u:[[1]] for u in [4,1,3,2,10,11]})
    bank=fit(h,[4,1,3,2],SETTINGS,n_items=2)
    users,q=build_vectors(h[h.user_id.isin([10,11])],SETTINGS,n_items=2)
    a,dist=find_neighbors(q,users,bank,k=2,batch_size=1)
    b,_=find_neighbors(q,users,bank,k=2,batch_size=10)
    assert bank["train_user_ids"][a].tolist()==[[1,2],[1,2]]
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(dist,0)


def test_rejects_target_baskets_duplicates_gaps_and_invalid_candidates():
    h=history({1:[[1],[2]],2:[[1],[1]]})
    with pytest.raises(ValueError,match="prior"):
        build_vectors(h.assign(eval_set="train"),SETTINGS,3)
    with pytest.raises(ValueError,match="at most once"):
        build_vectors(pd.concat([h,h.iloc[:1]]),SETTINGS,3)
    with pytest.raises(ValueError,match="contiguous"):
        build_vectors(h[h.order_number.eq(2)],SETTINGS,3)
    bank=fit(h,[1],SETTINGS,n_items=4)
    with pytest.raises(ValueError,match="personal history"):
        components(h[h.user_id.eq(2)],pd.DataFrame({"user_id":[2],"product_id":[3]}),bank,[1])
    with pytest.raises(ValueError,match="after excluding self"):
        users,q=build_vectors(h[h.user_id.eq(1)],SETTINGS,4)
        find_neighbors(q,users,bank,1)
