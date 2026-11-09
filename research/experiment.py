"""Bounded ablation study; selection on validation, confirmation after freezing."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from basketlab.metrics import attach_labels, evaluate
from .data import save_json, sha
from .models import fit, heuristic, predict, variants


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seeds(config, name):
    return [42] if heuristic(name) else config["model_seeds"]


def predictive_sources():
    return sorted(list(Path("basketlab").glob("*.py"))+[
        Path("research")/f for f in ["data.py","features.py","patterns.py","models.py",
                                     "experiment.py","analysis.py","protocol.json","run.py"]]
        +list(Path("tests").glob("*.py")))


def assert_hashes(entries):
    changed=[p for p,digest in entries.items() if not Path(p).exists() or sha(p)!=digest]
    if changed:raise ValueError(f"Frozen inputs changed: {changed}")


def aggregate(run, part, config):
    rows=[]
    for name in variants():
        tables=[]
        for seed in seeds(config,name):
            d=run/part/name/str(seed)
            info=load_json(d/"metrics.json")
            rows.append({"name":name,"seed":seed,**info})
            tables.append(pd.read_parquet(d/"users.parquet"))
        per=pd.concat(tables).groupby("user_id",as_index=False).mean(numeric_only=True)
        per.to_parquet(run/part/name/"users.parquet",index=False)
    all_seeds=pd.DataFrame(rows)
    all_seeds.to_csv(run/part/"seed_metrics.csv",index=False)
    summary=all_seeds.groupby("name",sort=False).mean(numeric_only=True).drop(columns="seed")
    summary["seed_recall_min"]=all_seeds.groupby("name").recall_at_10.min()
    summary["seed_recall_max"]=all_seeds.groupby("name").recall_at_10.max()
    summary.to_csv(run/part/"metrics.csv")
    return summary


def develop(config, cache, run):
    if (run/"freeze.json").exists():raise FileExistsError("Study frozen; development closed")
    run.mkdir(parents=True,exist_ok=True)
    inputs=predictive_sources()+[cache/"features.parquet",cache/"labels_train.parquet",
                                cache/"labels_valid.parquet",cache/"splits.csv"]
    digest={str(p):sha(p) for p in inputs}
    identity=run/"development_inputs.json"
    if identity.exists():
        if load_json(identity)!={"protocol":config,"sha256":digest}:raise ValueError("Development inputs changed")
    else:save_json(identity,{"protocol":config,"sha256":digest})
    all_features=pd.read_parquet(cache/"features.parquet").drop(columns="product_name")
    train=attach_labels(all_features[all_features.split.eq("train")],pd.read_parquet(cache/"labels_train.parquet"))
    valid=attach_labels(all_features[all_features.split.eq("valid")],pd.read_parquet(cache/"labels_valid.parquet"))
    del all_features
    for name in variants():
        for seed in seeds(config,name):
            dest=run/"validation"/name/str(seed);dest.mkdir(parents=True,exist_ok=True)
            if (dest/"metrics.json").exists():continue
            start=time.monotonic();artifact=fit(train,name,seed)
            scores=predict(valid,artifact)
            metrics,per,top=evaluate(valid,scores,k=10)
            metrics["fit_evaluate_seconds"]=time.monotonic()-start
            joblib.dump(artifact,dest/"model.joblib")
            per.to_parquet(dest/"users.parquet",index=False)
            top.to_parquet(dest/"top10.parquet",index=False)
            save_json(dest/"metrics.json",metrics)
            print(f"VALID {name} seed={seed}: {metrics['recall_at_10']:.6f} ({metrics['fit_evaluate_seconds']:.1f}s)",flush=True)
    summary=aggregate(run,"validation",config)
    selected=str(summary.recall_at_10.idxmax())
    cluster=str(summary.loc[[f"cluster{k}" for k in config["clustering_k"]]].recall_at_10.idxmax())
    window=str(summary.loc[["frequency","window3","window5","window10"]].recall_at_10.idxmax())
    selection={"selected":selected,"selected_cluster":cluster,"selected_window":window,
               "validation_recall":float(summary.loc[selected,"recall_at_10"]),
               "criterion":config["selection"],"metric_aggregation":"mean of per-seed metrics, not an ensemble"}
    save_json(run/"selection.json",selection)
    log=[];best=-1.
    for name,row in summary.iterrows():
        value=float(row.recall_at_10);keep=value>best
        log.append({"variant":name,"validation_recall":value,"status":"keep" if keep else "no_improvement",
                    "best_so_far":max(best,value),"note":"all predeclared variants retained for scientific ablation"})
        best=max(best,value)
    pd.DataFrame(log).to_csv(run/"research-results.tsv",sep="\t",index=False)
    return selection


def freeze(config, cache, run):
    if (run/"freeze.json").exists():raise FileExistsError("Freeze already exists")
    identity=load_json(run/"development_inputs.json")
    assert_hashes(identity["sha256"])
    if identity["protocol"]!=config:raise ValueError("Protocol changed")
    selection=load_json(run/"selection.json")
    inputs=predictive_sources()+list(cache.glob("*.parquet"))+list(cache.glob("*.json"))+[
        cache/"splits.csv",cache/"cluster_stability.csv",run/"selection.json",run/"development_inputs.json",
        run/"research-results.tsv",run/"validation/metrics.csv",run/"validation/seed_metrics.csv"]
    inputs+=list((run/"validation").rglob("model.joblib"))
    if len(list((run/"validation").rglob("model.joblib")))!=sum(len(seeds(config,n)) for n in variants()):
        raise ValueError("Incomplete model matrix")
    resolved=[[selection.get(a,a),selection.get(b,b)] for a,b in config["primary_contrasts"]]
    record={"utc":datetime.now(timezone.utc).isoformat(),"protocol":config,"selection":selection,
            "primary_contrasts":resolved,"sha256":{str(p):sha(p) for p in inputs},
            "python":platform.python_version(),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "note":"confirmation labels were partitioned and hashed, never used to fit or select models"}
    save_json(run/"freeze.json",record)
    return {"frozen_files":len(record["sha256"]),"contrasts":resolved,"selection":selection}


def confirm(config, cache, run):
    if (run/"confirmation/completed.json").exists():raise FileExistsError("Confirmation already completed")
    frozen=load_json(run/"freeze.json");assert_hashes(frozen["sha256"])
    if config!=frozen["protocol"]:raise ValueError("Protocol mismatch")
    dest=run/"confirmation";dest.mkdir(parents=True,exist_ok=True)
    if not (dest/"started.json").exists():
        save_json(dest/"started.json",{"utc":datetime.now(timezone.utc).isoformat(),"freeze_sha256":sha(run/"freeze.json")})
    if load_json(dest/"started.json")["freeze_sha256"]!=sha(run/"freeze.json"):raise ValueError("Freeze changed")
    frame=pd.read_parquet(cache/"features.parquet")
    frame=attach_labels(frame[frame.split.eq("confirm")],pd.read_parquet(cache/"labels_confirm.parquet"))
    for name in variants():
        for seed in seeds(config,name):
            d=dest/name/str(seed);d.mkdir(parents=True,exist_ok=True)
            if (d/"metrics.json").exists():continue
            artifact=joblib.load(run/"validation"/name/str(seed)/"model.joblib")
            scores=predict(frame,artifact);metrics,per,top=evaluate(frame,scores,10)
            per.to_parquet(d/"users.parquet",index=False);top.to_parquet(d/"top10.parquet",index=False)
            pd.DataFrame({"user_id":frame.user_id,"product_id":frame.product_id,"score":scores}).to_parquet(d/"scores.parquet",index=False)
            save_json(d/"metrics.json",metrics)
            print(f"CONFIRM {name} seed={seed}: {metrics['recall_at_10']:.6f}",flush=True)
    aggregate(run,"confirmation",config)
    from .analysis import analyze
    result=analyze(config,cache,run,frame,frozen)
    save_json(dest/"completed.json",{"utc":datetime.now(timezone.utc).isoformat(),"freeze_sha256":sha(run/"freeze.json"),"models":len(variants())})
    return result


def report(config,cache,run):
    from .reporting import create_report
    return create_report(config,cache,run)


def audit(config,cache,run):
    from .auditing import verify
    return verify(config,cache,run)
