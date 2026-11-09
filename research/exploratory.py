"""Post-confirmation diagnostic, added after covered-target gains were observed.

No fitting or model selection. Decomposes a fixed ranking's total Recall@10
change into covered/uncovered hit contributions using the same denominator.
This was NOT one of the eight preregistered primary tests.
"""
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import paired_interval
from .data import save_json,sha
from .experiment import assert_hashes,load_json


def main():
    run=Path("autoresearch/research-v2");cache=Path("data/processed/research_v2")
    freeze=load_json(run/"freeze.json");assert_hashes(freeze["sha256"])
    if not (run/"confirmation/completed.json").exists():raise RuntimeError("Main study must be complete")
    frame=pd.read_parquet(cache/"features.parquet")
    frame=frame[frame.split.eq("confirm")]
    target=pd.read_parquet(run/"confirmation/full/users.parquet").set_index("user_id").repeat_targets
    rows=[]
    for name in freeze["protocol"]["rule_settings"]:
        covered=frame[["user_id","product_id",name+"_covered"]].rename(columns={name+"_covered":"covered"})
        per={}
        for model in ["full",name]:
            parts=[]
            for seed in freeze["protocol"]["model_seeds"]:
                top=pd.read_parquet(run/"confirmation"/model/str(seed)/"top10.parquet").merge(covered,on=["user_id","product_id"],validate="one_to_one")
                top["covered_hits"]=top.label*top.covered
                top["uncovered_hits"]=top.label*(1-top.covered)
                per_seed=top.groupby("user_id")[["covered_hits","uncovered_hits","covered"]].sum()
                per_seed["covered_contribution"]=per_seed.covered_hits/target.replace(0,np.nan)
                per_seed["uncovered_contribution"]=per_seed.uncovered_hits/target.replace(0,np.nan)
                parts.append(per_seed.reset_index())
            per[model]=pd.concat(parts).groupby("user_id",as_index=False).mean(numeric_only=True)
        result={"rules":name,"covered_recommendations_per_user_delta":float(per[name].covered.mean()-per["full"].covered.mean())}
        for kind in ["covered","uncovered"]:
            left=per[name][["user_id",kind+"_contribution"]].rename(columns={kind+"_contribution":"recall"})
            right=per["full"][["user_id",kind+"_contribution"]].rename(columns={kind+"_contribution":"recall"})
            stats=paired_interval(left,right,family=6)
            result[kind+"_delta"]=stats["delta"]
            result[kind+"_ci95_low"]=stats["ci95_low"];result[kind+"_ci95_high"]=stats["ci95_high"]
        result["total_delta"]=result["covered_delta"]+result["uncovered_delta"]
        a=pd.read_parquet(run/"confirmation"/name/"users.parquet").set_index("user_id").recall
        b=pd.read_parquet(run/"confirmation/full/users.parquet").set_index("user_id").recall
        np.testing.assert_allclose(result["total_delta"],(a-b).mean(),atol=1e-12)
        rows.append(result)
    out=run/"exploratory";out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"rule_hit_decomposition.csv",index=False)
    save_json(out/"provenance.json",{"utc":datetime.now(timezone.utc).isoformat(),"status":"post-confirmation exploratory",
               "freeze_sha256":sha(run/"freeze.json"),"reason":"covered-target improvement coexists with no overall improvement",
               "no_new_models":True,"interpretation":"arithmetic decomposition of fixed rankings; not a causal mechanism test"})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":main()
