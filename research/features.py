from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from basketlab.enrichment import fit_clusters
from .patterns import pair_rules, score_rules
from .data import save_json


def build(config, cache):
    if (cache/"enriched.json").exists():raise FileExistsError("Feature protocol already prepared")
    c=pd.read_parquet(cache/"candidates.parquet")
    h=pd.read_parquet(cache/"history.parquet")
    users=pd.read_parquet(cache/"users.parquet")
    train_users=users.loc[users.split.eq("train"),"user_id"]
    pattern_info={}
    for name,settings in config["rule_settings"].items():
        print(f"Mining {name}",flush=True)
        rules,info=pair_rules(h,train_users,mode="sequential" if name=="sequential" else "co",**settings)
        f=score_rules(h,c,rules,name)
        c=c.merge(f,on=["user_id","product_id"],validate="one_to_one")
        rules.to_parquet(cache/f"rules_{name}.parquet",index=False)
        info["candidate_coverage_train"]=float(c.loc[c.split.eq("train"),f"{name}_covered"].mean())
        info["candidate_coverage_valid"]=float(c.loc[c.split.eq("valid"),f"{name}_covered"].mean())
        pattern_info[name]=info
    stability=[]
    for k in config["clustering_k"]:
        labels=[]
        for seed in config["clustering_seeds"]:
            f,profile,pipeline,sil=fit_clusters(users,n_clusters=k,seed=seed)
            labels.append(f.loc[users.split.eq("train"),"cluster"].to_numpy())
            stability.append({"k":k,"seed":seed,"silhouette":sil,"ari_vs_seed42":float(adjusted_rand_score(labels[0],labels[-1]))})
            if seed==42:
                renamed={x:f"k{k}_{x}" for x in f if x!="user_id"}
                c=c.merge(f.rename(columns=renamed),on="user_id",validate="many_to_one")
                profile.to_csv(cache/f"cluster_profiles_k{k}.csv")
                joblib.dump(pipeline,cache/f"cluster_k{k}.joblib")
        print(f"Clusters K={k} complete",flush=True)
    pd.DataFrame(stability).to_csv(cache/"cluster_stability.csv",index=False)
    c.to_parquet(cache/"features.parquet",index=False)
    metadata={"patterns":pattern_info,"clustering_stability":stability,"feature_rows":len(c)}
    save_json(cache/"enriched.json",metadata)
    return metadata
