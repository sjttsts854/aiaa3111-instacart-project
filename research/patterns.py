from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


RULE_COLUMNS=["antecedent","consequent","count","support","confidence","lift"]


def pair_rules(history, train_users, mode="co", top_n=500, min_support=.002,
               min_count=30, min_confidence=0., min_lift=1.1, max_outgoing=None):
    """Exact length-two mining by sparse counts. Sequential rows never cross users."""
    if mode not in {"co","sequential"}:raise ValueError("Unknown pattern mode")
    h=history[history.user_id.isin(train_users)]
    if h.duplicated(["order_id","product_id"]).any():raise ValueError("Nonbinary basket")
    orders=h[["order_id","user_id","order_number"]].drop_duplicates().sort_values(["user_id","order_number"])
    ids=pd.Index(orders.order_id)
    top=h.product_id.value_counts().head(top_n).index.sort_values()
    kept=h[h.product_id.isin(top)]
    x=coo_matrix((np.ones(len(kept),dtype=np.int32),
                  (ids.get_indexer(kept.order_id),top.get_indexer(kept.product_id))),
                 shape=(len(ids),len(top))).tocsr()
    if mode=="co":
        left=right=x; transitions=len(orders)
    else:
        users=orders.user_id.to_numpy(); numbers=orders.order_number.to_numpy()
        valid=(users[:-1]==users[1:]) & (numbers[1:]==numbers[:-1]+1)
        left=x[np.flatnonzero(valid)];right=x[np.flatnonzero(valid)+1]
        transitions=int(valid.sum())
    if not transitions:return pd.DataFrame(columns=RULE_COLUMNS),{"contexts":0,"top_products":len(top),"rules":0}
    counts=(left.T@right).tocoo()
    antecedent_n=np.asarray(left.sum(axis=0)).ravel()
    consequent_n=np.asarray(right.sum(axis=0)).ravel()
    mask=(counts.data>=max(min_count,min_support*transitions))&(counts.row!=counts.col)
    a,b,n=counts.row[mask],counts.col[mask],counts.data[mask]
    confidence=n/antecedent_n[a];lift=confidence/(consequent_n[b]/transitions)
    mask=(confidence>=min_confidence)&(lift>=min_lift)
    rules=pd.DataFrame({"antecedent":top.to_numpy()[a[mask]],"consequent":top.to_numpy()[b[mask]],
                        "count":n[mask],"support":n[mask]/transitions,"confidence":confidence[mask],"lift":lift[mask]})
    rules=rules.sort_values(["antecedent","confidence","count","consequent"],ascending=[True,False,False,True])
    if max_outgoing is not None:rules=rules.groupby("antecedent",sort=False).head(max_outgoing)
    return rules.reset_index(drop=True),{"contexts":transitions,"top_products":len(top),"rules":len(rules),
                                         "mode":mode,"self_rules":False,"fit_users":len(set(train_users))}


def score_rules(history, candidates, rules, prefix):
    """Only observed last baskets trigger rules. Keep the full candidate pool."""
    cols=[f"{prefix}_{x}" for x in ["confidence_sum","confidence_mean","lift_mean","hits","covered"]]
    out=candidates[["user_id","product_id"]].copy()
    if rules.empty:
        for name in cols:out[name]=0.
        return out
    items=pd.Index(np.sort(np.union1d(rules.antecedent,rules.consequent)))
    users=pd.Index(np.sort(candidates.user_id.unique()))
    last=history[history.age.eq(0)&history.product_id.isin(items)]
    x=coo_matrix((np.ones(len(last),dtype=np.float32),
                  (users.get_indexer(last.user_id),items.get_indexer(last.product_id))),
                 shape=(len(users),len(items))).tocsr()
    cr=users.get_indexer(candidates.user_id);cc=items.get_indexer(candidates.product_id)
    known=cc>=0
    rows=items.get_indexer(rules.antecedent);columns=items.get_indexer(rules.consequent)
    def gather(values):
        relation=coo_matrix((np.asarray(values,dtype=np.float32),(rows,columns)),shape=(len(items),len(items))).tocsr()
        scores=x@relation
        result=np.zeros(len(candidates),dtype=np.float32)
        result[known]=np.asarray(scores[cr[known],cc[known]]).ravel()
        return result
    hits=gather(np.ones(len(rules)))
    conf=gather(rules.confidence)
    lift=gather(rules.lift)
    out[cols[0]]=conf;out[cols[1]]=conf/np.maximum(1,hits)
    out[cols[2]]=lift/np.maximum(1,hits);out[cols[3]]=hits;out[cols[4]]=(hits>0).astype(np.int8)
    return out
