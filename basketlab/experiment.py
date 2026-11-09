from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .metrics import attach_labels, evaluate, paired_bootstrap
from .models import fit_predict, predict_saved


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def verify(cache: Path, output: Path, config: dict):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(cache / "features.parquet")
    train = attach_labels(frame[frame.split.eq("train")], pd.read_parquet(cache / "labels_train.parquet"))
    valid = attach_labels(frame[frame.split.eq("valid")], pd.read_parquet(cache / "labels_valid.parquet"))
    scores, model, columns, elapsed = fit_predict(train, valid, config)
    summary, per_user, top = evaluate(valid, scores)
    summary.update({"fit_and_predict_seconds": elapsed, "candidate_rows_train": len(train),
                    "candidate_rows_valid": len(valid), "config": config,
                    "feature_columns": columns, "split": "valid"})
    write_json(output / "metrics.json", summary)
    per_user.to_csv(output / "per_user.csv", index=False)
    top.to_parquet(output / "top_predictions.parquet", index=False)
    joblib.dump({"model": model, "columns": columns, "config": config}, output / "model.joblib")
    if model is not None and hasattr(model, "feature_importances_"):
        pd.DataFrame({"feature": columns, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    return summary


def finalize(cache: Path, run: Path):
    """One-shot test evaluation of the frozen, predeclared comparison matrix."""
    destination = run / "test"
    if destination.exists():
        raise FileExistsError("Test evaluation already exists. Do not repeatedly tune against this holdout.")
    selected = json.loads((run / "selected.json").read_text(encoding="utf-8"))
    plan = json.loads((run / "experiment_plan.json").read_text(encoding="utf-8"))
    features = pd.read_parquet(cache / "features.parquet")
    frame = features[features.split.eq("test")].copy()
    labels = pd.read_parquet(cache / "labels_test.parquet")
    test = attach_labels(frame, labels)
    destination.mkdir()
    write_json(destination / "frozen_selection.json", selected)
    records, users = [], {}
    for entry in plan:
        name = entry["name"]
        artifact_path = run / "validation" / name / "model.joblib"
        if not artifact_path.exists():
            continue
        artifact = joblib.load(artifact_path)
        start = time.perf_counter()
        scores = predict_saved(test, artifact)
        elapsed = time.perf_counter()-start
        summary, per, top = evaluate(test, scores)
        summary.update({"name": name, "predict_seconds": elapsed,
                        "ms_per_user": elapsed/len(per)*1000,
                        "selected_on_validation": name==selected["name"]})
        records.append(summary); users[name] = per
        per.to_csv(destination / f"{name}_users.csv", index=False)
        if name==selected["name"]:
            top.to_parquet(destination / "selected_predictions.parquet", index=False)
    results = pd.DataFrame(records)
    results.to_csv(destination / "metrics.csv", index=False)
    comparisons = {}
    baseline = "personal_frequency"
    for name, per in users.items():
        if name != baseline:
            comparisons[f"{name}_minus_{baseline}"] = paired_bootstrap(per, users[baseline])
    for name in ["lightgbm_clusters", "lightgbm_rules", "lightgbm_both"]:
        if name in users and "lightgbm_base" in users:
            comparisons[f"{name}_minus_lightgbm_base"] = paired_bootstrap(users[name], users["lightgbm_base"])
    write_json(destination / "bootstrap.json", comparisons)
    profiles = pd.read_parquet(cache / "users.parquet")
    train_profiles = profiles[profiles.split.eq("train")]
    cut = float(train_profiles.historical_repeat_rate.median())
    profiles["history_group"] = pd.cut(profiles.history_orders, bins=[0,5,15,100], labels=["3-5","6-15","16+"])
    profiles["habit_group"] = np.where(profiles.historical_repeat_rate>=cut, "higher_repeat", "lower_repeat")
    subgroups = []
    for name, per in users.items():
        joined = per.merge(profiles, on="user_id", validate="one_to_one")
        for dimension in ["history_group", "habit_group"]:
            for group, part in joined.groupby(dimension, observed=True):
                subgroups.append({"model": name, "dimension": dimension, "group": str(group),
                                  "users": len(part), "users_with_repeat_targets": int(part.recall.notna().sum()),
                                  "recall_at_10": float(part.recall.mean()), "ndcg_at_10": float(part.ndcg.mean())})
    pd.DataFrame(subgroups).to_csv(destination / "subgroups.csv", index=False)
    n_all = labels.groupby("user_id").size()
    n_rep = test.groupby("user_id").label.sum()
    write_json(destination / "scope.json", {
        "users": len(n_all), "candidate_scope": "previously purchased products only",
        "mean_repeat_fraction_of_full_target_basket": float((n_rep/n_all).mean()),
        "overall_repeat_fraction": float(n_rep.sum()/n_all.sum()),
        "habit_group_threshold_fitted_on_train": cut,
        "interpretation": "Primary recall denominator is repeat targets, not the complete next basket.",
        "python": platform.python_version()})
    make_figures(run, results)
    # Evidence hash makes future modifications of the test report detectable.
    write_json(destination / "completed.json", {"selection": selected["name"],
        "metrics_sha256": hashlib.sha256((destination / "metrics.csv").read_bytes()).hexdigest()})
    return results.to_dict(orient="records")


def make_figures(run, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = run / "figures"; out.mkdir(exist_ok=True)
    ordered = results.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9,4.8))
    colors = ["#087e8b" if x else "#9eabb5" for x in ordered.selected_on_validation]
    ax.barh(ordered.name, ordered.recall_at_10, color=colors)
    ax.set(xlabel="Macro Recall@10 on non-empty repeat targets", title="Held-out test comparison")
    ax.set_xlim(0, min(1., float(results.recall_at_10.max())+.1))
    for i, val in enumerate(ordered.recall_at_10): ax.text(val+.005,i,f"{val:.3f}",va="center")
    ax.spines[["right","top"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "test_recall.png",dpi=180); plt.close(fig)
    sub = pd.read_csv(run / "test" / "subgroups.csv")
    sub = sub[(sub.dimension=="history_group") & sub.model.isin(["personal_frequency","lightgbm_base","lightgbm_both"])]
    pivot = sub.pivot(index="group",columns="model",values="recall_at_10").reindex(["3-5","6-15","16+"])
    ax = pivot.plot.bar(figsize=(8,4.5),rot=0)
    ax.set(xlabel="Number of observed historical orders",ylabel="Recall@10",title="Performance by history length")
    ax.figure.tight_layout();ax.figure.savefig(out / "history_subgroups.png",dpi=180);plt.close(ax.figure)
