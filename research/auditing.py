"""Independent checks use raw target files and set operations, not evaluate()."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import save_json,sha
from .experiment import assert_hashes,load_json


def verify(config,cache,run):
    frozen=load_json(run/"freeze.json");assert_hashes(frozen["sha256"])
    validation=pd.read_csv(run/"validation/metrics.csv").set_index("name")
    assert frozen["selection"]["selected"]==validation.recall_at_10.idxmax()
    declared=frozen["primary_contrasts"]
    contrasts=pd.read_csv(run/"confirmation/primary_contrasts.csv")
    assert declared==contrasts[["left","right"]].values.tolist()
    for row in contrasts.itertuples():
        left=pd.read_parquet(run/"confirmation"/row.left/"users.parquet").set_index("user_id").recall
        right=pd.read_parquet(run/"confirmation"/row.right/"users.parquet").set_index("user_id").recall
        np.testing.assert_allclose((left-right).mean(),row.delta,atol=1e-12)
    if load_json(run/"confirmation/completed.json")["freeze_sha256"]!=sha(run/"freeze.json"):
        raise AssertionError("Completion points to another freeze")
    manifest=load_json(cache/"manifest.json")
    for name,info in manifest["source"].items():
        assert sha(Path("archive")/name)==info["sha256"],name
    splits=pd.read_csv(cache/"splits.csv")
    assert splits.user_id.nunique()==len(splits)==33000
    old=pd.read_csv(config["excluded_users"])
    assert not splits.user_id.isin(old.user_id).any()
    assert splits.split.value_counts().to_dict()=={"train":config["train_users"],"confirm":config["confirmation_users"],"valid":config["validation_users"]}
    ids=set(splits.loc[splits.split.eq("confirm"),"user_id"])
    orders=pd.read_csv("archive/orders.csv")
    prior_orders=orders[orders.user_id.isin(ids)&orders.eval_set.eq("prior")]
    targets=orders[orders.user_id.isin(ids)&orders.eval_set.eq("train")]
    maxhist=prior_orders.groupby("user_id").order_number.max()
    assert (targets.order_number.to_numpy()==targets.user_id.map(maxhist).to_numpy()+1).all()
    raw=pd.read_csv("archive/order_products__train.csv").merge(targets[["order_id","user_id"]],on="order_id")
    saved=pd.read_parquet(cache/"labels_confirm.parquet")
    key=["user_id","product_id"]
    pd.testing.assert_frame_equal(raw[key].sort_values(key).reset_index(drop=True),saved[key].sort_values(key).reset_index(drop=True))
    # Entire confirmation candidate universe is independently reconstructed from raw CSV.
    basket_ids=set(prior_orders.order_id)
    parts=[chunk[chunk.order_id.isin(basket_ids)][["order_id","product_id"]]
           for chunk in pd.read_csv("archive/order_products__prior.csv",chunksize=1000000)]
    prior=pd.concat(parts,ignore_index=True).merge(prior_orders[["order_id","user_id"]],on="order_id")
    truth_candidates=prior[key].drop_duplicates().sort_values(key).reset_index(drop=True)
    features=pd.read_parquet(cache/"features.parquet")
    stored=features[features.split.eq("confirm")][key].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(truth_candidates,stored)
    raw_targets=raw.groupby("user_id").product_id.agg(set).to_dict()
    pools=truth_candidates.groupby("user_id").product_id.agg(set).to_dict()
    repeats={u:raw_targets[u]&pools[u] for u in ids}
    checked=[]
    for name in sorted(set(["frequency","full",frozen["selection"]["selected"]])):
        for seed in ([42] if name=="frequency" else config["model_seeds"]):
            d=run/"confirmation"/name/str(seed)
            if not d.exists():continue
            top=pd.read_parquet(d/"top10.parquet")
            assert not top.duplicated(key).any()
            score=pd.read_parquet(d/"scores.parquet").sort_values(["user_id","score","product_id"],ascending=[True,False,True])
            manual_top=score.groupby("user_id",sort=False).head(10)
            pd.testing.assert_frame_equal(top[key].reset_index(drop=True),manual_top[key].reset_index(drop=True))
            scores=[]
            for u,group in top.groupby("user_id",sort=False):
                items=group.product_id.to_list();hit=[int(p in repeats[u]) for p in items]
                count=len(repeats[u]);hits=sum(hit)
                scores.append({"user_id":u,"recall":hits/count if count else np.nan,
                               "precision":hits/len(items),"f1":2*hits/(count+len(items)),
                               "ndcg":sum(h/np.log2(r+2) for r,h in enumerate(hit))/sum(1/np.log2(r+2) for r in range(min(count,10))) if count else np.nan})
            independent=pd.DataFrame(scores).set_index("user_id")
            saved_per=pd.read_parquet(d/"users.parquet").set_index("user_id")
            for metric in ["recall","precision","f1","ndcg"]:
                np.testing.assert_allclose(independent[metric].sort_index(),saved_per[metric].sort_index(),atol=1e-12,equal_nan=True)
            recorded=load_json(d/"metrics.json")
            for metric in ["recall","precision","f1","ndcg"]:
                np.testing.assert_allclose(independent[metric].mean(),recorded[f"{metric}_at_10"],atol=1e-12)
            checked.append({"model":name,"seed":seed,"independent_recall":float(independent.recall.mean())})
    report={"status":"PASS","frozen_files":len(frozen["sha256"]),"raw_sources":len(manifest["source"]),
            "previous_user_overlap":0,"confirmation_users":len(ids),"candidate_rows_checked":len(stored),
            "raw_target_rows_checked":len(raw),"independent_metrics":checked,
            "scope":"raw CSV target/candidate reconstruction, target chronology, ranking and four per-user metrics; frozen hashes"}
    save_json(run/"audit.json",report)
    return report
