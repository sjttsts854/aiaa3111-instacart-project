from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier

FREQUENCY=["purchase_count","frequency","history_orders","basket_mean","basket_std",
           "distinct_products","historical_repeat_rate","cart_position_mean","single_purchase"]
METADATA=["aisle_id","department_id","dept_affinity"]
RECENCY=["recent3_count","recent5_count","decay_score","orders_since_purchase","in_last_basket","frequency_since_first"]
CADENCE=["mean_purchase_gap","cadence_std","cadence_cv","interval_mean","interval_std"]
INTERACTION=["gap_ratio"]
GLOBAL=["global_count","global_repeat_rate","global_support","unseen_in_training"]


def variants():
    return ["frequency","decay","window3","window5","window10","frequency_context", "frequency_recency",
            "no_temporal","no_cadence","no_global","full","co_narrow","co_wide","sequential",
            "cluster3","cluster6","cluster10"]


def columns(name, frame):
    core=FREQUENCY+METADATA
    if name=="frequency_context":return core
    if name=="frequency_recency":return core+RECENCY
    if name=="no_temporal":return core+GLOBAL
    if name=="no_cadence":return core+RECENCY+GLOBAL
    if name=="no_global":return core+RECENCY+CADENCE+INTERACTION
    full=core+RECENCY+CADENCE+INTERACTION+GLOBAL
    if name in {"co_narrow","co_wide","sequential"}:
        return full+[x for x in frame if x.startswith(name+"_")]
    if name.startswith("cluster"):
        prefix=f"k{name.replace('cluster','')}_cluster_"
        return full+[x for x in frame if x.startswith(prefix)]
    if name=="full":return full
    raise ValueError(name)


def heuristic(name):return name in {"frequency","decay","window3","window5","window10"}


def predict(frame, artifact):
    name=artifact["name"]
    if heuristic(name):
        col={"frequency":"frequency","decay":"decay_score","window3":"recent3_count",
             "window5":"recent5_count","window10":"recent10_count"}[name]
        return frame[col].to_numpy()
    return artifact["model"].predict_proba(frame[artifact["columns"]].replace([np.inf,-np.inf],np.nan))[:,1]


def fit(train, name, seed):
    artifact={"name":name,"seed":seed,"model":None,"columns":[]}
    if heuristic(name):return artifact
    cols=columns(name,train)
    # Random feature subsampling makes seed sensitivity a real robustness check.
    # These settings are shared by every learned ablation; no per-model tuning.
    model=LGBMClassifier(n_estimators=240,learning_rate=.05,num_leaves=31,min_child_samples=80,
                          colsample_bytree=.9,reg_lambda=1.,n_jobs=4,random_state=seed,
                          deterministic=True,force_col_wise=True,verbosity=-1)
    model.fit(train[cols].replace([np.inf,-np.inf],np.nan),train.label,
              categorical_feature=["aisle_id","department_id"])
    artifact.update(model=model,columns=cols)
    return artifact
