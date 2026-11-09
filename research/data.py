from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from basketlab.data import build_candidates


def sha(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):d.update(b)
    return d.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def fresh_split(eligible, excluded, config):
    pool=np.setdiff1d(eligible,excluded)
    counts=[config["train_users"],config["validation_users"],config["confirmation_users"]]
    if len(pool)<sum(counts):raise ValueError("Not enough previously unused users")
    ids=np.random.default_rng(config["split_seed"]).permutation(pool)[:sum(counts)]
    split=np.repeat(["train","valid","confirm"],counts)
    return pd.DataFrame({"user_id":ids,"split":split}).sort_values("user_id")


def extra_history_features(candidates, history):
    h=history.sort_values(["user_id","product_id","order_number"])
    gap=h.groupby(["user_id","product_id"]).order_number.diff()
    h=h.assign(gap=gap)
    stats=h.groupby(["user_id","product_id"]).gap.agg(["mean","std"]).reset_index()
    stats["cadence_cv"]=stats["std"]/stats["mean"]
    stats=stats.rename(columns={"std":"cadence_std"})
    c=candidates.merge(stats[["user_id","product_id","cadence_cv","cadence_std"]],
                       on=["user_id","product_id"],validate="one_to_one")
    counts=h[h.age.lt(10)].groupby(["user_id","product_id"]).size().rename("recent10_count").reset_index()
    c=c.merge(counts,on=["user_id","product_id"],how="left",validate="one_to_one")
    c["recent10_count"]=c.recent10_count.fillna(0)
    return c


def prepare(config, out: Path, raw=Path("archive")):
    if (out/"manifest.json").exists():raise FileExistsError("Prepared study exists")
    out.mkdir(parents=True,exist_ok=True)
    excluded=pd.read_csv(config["excluded_users"]).user_id
    orders=pd.read_csv(raw/"orders.csv")
    splits=fresh_split(orders.loc[orders.eval_set.eq("train"),"user_id"],excluded,config)
    selected=orders[orders.user_id.isin(splits.user_id)]
    prior_orders=selected[selected.eval_set.eq("prior")].copy()
    terminal=selected[selected.eval_set.eq("train")][["order_id","user_id"]]
    history=pd.concat([c[c.order_id.isin(prior_orders.order_id)] for c in pd.read_csv(
        raw/"order_products__prior.csv",chunksize=1000000)],ignore_index=True)
    products=pd.read_csv(raw/"products.csv")
    train_users=splits.loc[splits.split.eq("train"),"user_id"]
    candidates,users,h=build_candidates(history,prior_orders,products,train_users)
    candidates=extra_history_features(candidates,h).merge(splits,on="user_id",validate="many_to_one")
    users=users.merge(splits,on="user_id",validate="one_to_one")
    splits.to_csv(out/"splits.csv",index=False)
    candidates.to_parquet(out/"candidates.parquet",index=False)
    users.to_parquet(out/"users.parquet",index=False)
    h.to_parquet(out/"history.parquet",index=False)
    products.to_parquet(out/"products.parquet",index=False)
    labels=pd.read_csv(raw/"order_products__train.csv").merge(terminal,on="order_id",validate="many_to_one")
    for part in ["train","valid","confirm"]:
        rows=labels[labels.user_id.isin(splits.loc[splits.split.eq(part),"user_id"])]
        if rows.user_id.nunique()!=int(splits.split.eq(part).sum()):raise ValueError("Missing label baskets")
        rows[["user_id","product_id"]].to_parquet(out/f"labels_{part}.parquet",index=False)
    source={p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in raw.glob("*.csv")}
    manifest={"users":len(splits),"split_counts":splits.split.value_counts().to_dict(),
              "old_user_overlap":int(splits.user_id.isin(excluded).sum()),
              "history_rows":len(h),"history_baskets":int(h.order_id.nunique()),
              "candidate_rows":len(candidates),"source":source,
              "labels_note":"confirmation labels automatically partitioned but not inspected or used during development"}
    save_json(out/"manifest.json",manifest)
    return manifest
