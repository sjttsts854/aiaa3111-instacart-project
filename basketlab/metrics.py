from __future__ import annotations

import numpy as np
import pandas as pd


def attach_labels(candidates, targets):
    if targets.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Duplicate target products")
    out = candidates.merge(targets.assign(label=1), on=["user_id", "product_id"],
                           how="left", validate="one_to_one")
    out["label"] = out.label.fillna(0).astype("int8")
    return out


def evaluate(frame, scores, k=10):
    """Macro per-user metrics. Empty repeat-target users remain in the report."""
    if frame.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Duplicate candidates")
    if k < 1 or len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("Invalid scores or K")
    ranked = frame[["user_id", "product_id", "label"]].copy()
    ranked["score"] = scores
    ranked = ranked.sort_values(["user_id", "score", "product_id"], ascending=[True, False, True])
    ranked["rank"] = ranked.groupby("user_id").cumcount()+1
    top = ranked[ranked["rank"]<=k].copy()
    top["gain"] = top.label / np.log2(top["rank"]+1)
    per = ranked.groupby("user_id").label.sum().rename("repeat_targets").to_frame()
    per["hits"] = top.groupby("user_id").label.sum()
    per["recommended"] = top.groupby("user_id").size()
    per["recall"] = per.hits / per.repeat_targets.replace(0, np.nan)
    per["precision"] = per.hits / per.recommended
    per["f1"] = 2*per.hits / (per.repeat_targets+per.recommended)
    discounts = np.r_[0., np.cumsum(1/np.log2(np.arange(1,k+1)+1))]
    per["ndcg"] = top.groupby("user_id").gain.sum() / np.where(
        per.repeat_targets>0, discounts[np.minimum(k, per.repeat_targets).astype(int)], np.nan)
    metrics = {f"recall_at_{k}": float(per.recall.mean()), f"ndcg_at_{k}": float(per.ndcg.mean()),
               f"precision_at_{k}": float(per.precision.mean()), f"f1_at_{k}": float(per.f1.mean()),
               "users": len(per), "empty_repeat_targets": int(per.repeat_targets.eq(0).sum()),
               "mean_recommendation_length": float(per.recommended.mean())}
    return metrics, per.reset_index(), top.drop(columns="gain")


def paired_bootstrap(left, right, seed=42, n_boot=1000):
    pair = left[["user_id", "recall"]].merge(right[["user_id", "recall"]], on="user_id", suffixes=("_a","_b"))
    delta = (pair.recall_a-pair.recall_b).dropna().to_numpy()
    if not len(delta):
        return {"users": 0, "mean_delta": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(n_boot)])
    return {"users": len(delta), "mean_delta": float(delta.mean()),
            "ci95": [float(x) for x in np.quantile(samples,[.025,.975])]}
