"""Predeclared statistics plus explicitly exploratory diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from basketlab.metrics import evaluate
from .data import save_json
from .models import heuristic


def paired_interval(left,right,n_boot=6000,seed=42,family=8):
    pair=left[["user_id","recall"]].merge(right[["user_id","recall"]],on="user_id",suffixes=("_a","_b"),validate="one_to_one")
    delta=(pair.recall_a-pair.recall_b).dropna().to_numpy()
    if not len(delta):return {"users":0,"delta":None,"ci95_low":None,"ci95_high":None,"sim_low":None,"sim_high":None}
    rng=np.random.default_rng(seed);boot=np.empty(n_boot)
    for start in range(0,n_boot,100):
        size=min(100,n_boot-start)
        boot[start:start+size]=delta[rng.integers(len(delta),size=(size,len(delta)))].mean(axis=1)
    q=np.quantile(boot,[.025,.975,.05/(2*family),1-.05/(2*family)])
    return {"users":len(delta),"delta":float(delta.mean()),"ci95_low":float(q[0]),"ci95_high":float(q[1]),
            "sim_low":float(q[2]),"sim_high":float(q[3])}


def common_support(frame,minimum):
    """Pooled cell weights, identical for every history group; descriptive only."""
    df=frame.dropna(subset=["recall","candidate_bin","target_bin"]).copy()
    groups=["3-5","6-15","16+"]
    counts=df.groupby(["candidate_bin","target_bin","history_group"],observed=True).size().unstack("history_group",fill_value=0).reindex(columns=groups,fill_value=0)
    eligible=counts.index[(counts>=minimum).all(axis=1)]
    df=df.set_index(["candidate_bin","target_bin"])
    included=df.index.isin(eligible);supported=df[included].reset_index()
    if supported.empty:return pd.DataFrame(),{"cells":0,"included_users":0,"nonempty_users":len(df)}
    weights=supported.groupby(["candidate_bin","target_bin"],observed=True).size()
    weights=weights/weights.sum()
    rows=[]
    for group in groups:
        sub=supported[supported.history_group.eq(group)]
        means=sub.groupby(["candidate_bin","target_bin"],observed=True).recall.mean()
        rows.append({"history_group":group,"standardized_recall":float((means*weights).sum()),
                     "common_support_users":len(sub),"raw_common_support_recall":float(sub.recall.mean())})
    return pd.DataFrame(rows),{"cells":len(eligible),"included_users":len(supported),"nonempty_users":len(df),
                               "cell_weights":[{"candidate_bin":str(a),"target_bin":str(b),"weight":float(w)} for (a,b),w in weights.items()]}


def analyze(config,cache,run,frame,frozen):
    dest=run/"confirmation";primary=[]
    def per(name):return pd.read_parquet(dest/name/"users.parquet")
    for a,b in frozen["primary_contrasts"]:
        r={"left":a,"right":b,**paired_interval(per(a),per(b),config["bootstrap_replicates"],family=len(frozen["primary_contrasts"]))}
        r["meaningful_positive"]=r["sim_low"]>config["meaningful_improvement"]
        r["meaningful_gain_ruled_out"]=r["sim_high"]<config["meaningful_improvement"]
        primary.append(r)
    pd.DataFrame(primary).to_csv(dest/"primary_contrasts.csv",index=False)
    base=per("full")
    info=frame.groupby("user_id").agg(candidates=("product_id","size"),history_orders=("history_orders","first"))
    details=base.merge(info,on="user_id",validate="one_to_one")
    details["history_group"]=pd.cut(details.history_orders,[2,5,15,100],labels=["3-5","6-15","16+"])
    details["candidate_bin"]=pd.cut(details.candidates,config["history_diagnostics"]["candidate_bins"])
    details["target_bin"]=pd.cut(details.repeat_targets,config["history_diagnostics"]["target_bins"])
    details["oracle_recall"]=np.minimum(10,details.repeat_targets)/details.repeat_targets.replace(0,np.nan)
    groups=details.groupby("history_group",observed=True).agg(users=("user_id","size"),nonempty=("recall","count"),
              recall=("recall","mean"),candidate_mean=("candidates","mean"),candidate_median=("candidates","median"),
              target_mean=("repeat_targets","mean"),oracle_recall=("oracle_recall","mean"))
    groups.to_csv(dest/"history_raw.csv")
    standardized,support=common_support(details,config["history_diagnostics"]["common_support_min_per_history_group"])
    standardized.to_csv(dest/"history_standardized.csv",index=False)
    save_json(dest/"history_common_support.json",support)
    details.drop(columns=["candidate_bin","target_bin"]).to_parquet(dest/"history_users.parquet",index=False)
    windows=[]
    long_ids=details.loc[details.history_orders.ge(16),"user_id"]
    for name in ["window3","window5","window10","decay"]:
        r=paired_interval(per(name)[lambda x:x.user_id.isin(long_ids)],per("frequency")[lambda x:x.user_id.isin(long_ids)],config["bootstrap_replicates"],family=4)
        windows.append({"model":name,"reference":"frequency","subgroup":"history>=16",**r})
    pd.DataFrame(windows).to_csv(dest/"history_window_comparisons.csv",index=False)
    diagnostics=[];strata=[]
    for name in config["rule_settings"]:
        covered=frame[f"{name}_covered"].gt(0)
        subset=frame.loc[covered,["user_id","product_id","label"]]
        denominator=subset.groupby("user_id").label.sum().rename("covered_targets")
        totals=frame.groupby("user_id").label.sum()
        conditional={};seed_lists=config["model_seeds"]
        for model in ["full",name]:
            tables=[]
            for seed in seed_lists:
                top=pd.read_parquet(dest/model/str(seed)/"top10.parquet")
                hits=top.merge(subset,on=["user_id","product_id"],suffixes=("","_checked")).groupby("user_id").label.sum()
                tab=denominator.to_frame();tab["recall"]=hits.reindex(tab.index,fill_value=0)/tab.covered_targets.replace(0,np.nan)
                tables.append(tab.reset_index())
            conditional[model]=pd.concat(tables).groupby("user_id",as_index=False).mean(numeric_only=True)
        delta=paired_interval(conditional[name],conditional["full"],config["bootstrap_replicates"],family=3)
        diagnostics.append({"rules":name,"candidate_coverage":float(covered.mean()),
            "target_coverage_micro":float(subset.label.sum()/frame.label.sum()),
            "target_coverage_macro":float((denominator.reindex(totals.index,fill_value=0)/totals.replace(0,np.nan)).mean()),
            "users_any_covered_target":int(denominator.gt(0).sum()),
            "full_covered_recall":float(conditional["full"].recall.mean()),
            "rule_covered_recall":float(conditional[name].recall.mean()),**delta})
        uc=frame.assign(covered=covered).groupby("user_id").covered.mean().rename("coverage")
        sub=per(name).merge(per("full")[["user_id","recall"]],on="user_id",suffixes=("","_base")).merge(uc,on="user_id")
        sub["stratum"]=pd.cut(sub.coverage,[-.001,0,.1,.3,1],labels=["zero","0-10%","10-30%","30-100%"])
        for group,part in sub.groupby("stratum",observed=True):
            strata.append({"rules":name,"stratum":str(group),"users":len(part),"nonempty_users":int(part.recall.notna().sum()),
                           "delta":float((part.recall-part.recall_base).mean())})
    pd.DataFrame(diagnostics).to_csv(dest/"rule_diagnostics.csv",index=False)
    pd.DataFrame(strata).to_csv(dest/"rule_coverage_strata.csv",index=False)
    sensitivities=[]
    for name in sorted(set(["frequency","decay","full",frozen["selection"]["selected"]])):
        for seed in ([42] if heuristic(name) else config["model_seeds"]):
            scores=pd.read_parquet(dest/name/str(seed)/"scores.parquet")
            aligned=frame.merge(scores,on=["user_id","product_id"],validate="one_to_one")
            for k in config["sensitivity_k"]:
                metrics,_,_=evaluate(aligned,aligned.score.to_numpy(),k)
                sensitivities.append({"name":name,"seed":seed,"k":k,**metrics})
    pd.DataFrame(sensitivities).to_csv(dest/"k_sensitivity.csv",index=False)
    target_size=pd.read_parquet(cache/"labels_confirm.parquet").groupby("user_id").size().rename("all_targets")
    full=base.merge(target_size,on="user_id",validate="one_to_one")
    scope={"users":len(full),"empty_repeat_targets":int(full.repeat_targets.eq(0).sum()),
           "candidate_recall_ceiling_macro":float((full.repeat_targets/full.all_targets).mean()),
           "candidate_recall_ceiling_micro":float(full.repeat_targets.sum()/full.all_targets.sum()),
           "full_model_whole_basket_recall_at_10":float((full.hits/full.all_targets).mean()),
           "full_model_repeat_recall_at_10":float(full.recall.mean()),
           "note":"primary metric excludes empty repeat targets; whole-basket recall includes all users"}
    save_json(dest/"scope.json",scope)
    return {"primary_contrasts":primary,"scope":scope,"history_common_support":support}
