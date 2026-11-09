from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def split_users(users, sample_size=10000, seed=42):
    """Sample complete users and assign disjoint 70/15/15 groups."""
    users = np.sort(np.unique(users))
    if not 20 <= sample_size <= len(users):
        raise ValueError(f"sample_size must be between 20 and {len(users)}")
    chosen = np.random.default_rng(seed).permutation(users)[:sample_size]
    n_train, n_valid = int(sample_size * .7), int(sample_size * .15)
    split = np.array(["train"] * n_train + ["valid"] * n_valid
                     + ["test"] * (sample_size - n_train - n_valid))
    return pd.DataFrame({"user_id": chosen, "split": split}).sort_values("user_id")


def build_candidates(history, orders, products, train_users):
    """Only prior history and static catalogue metadata are accepted here."""
    if not orders.eval_set.eq("prior").all():
        raise ValueError("Feature construction accepts prior orders only")
    h = history.merge(orders, on="order_id", validate="many_to_one")
    if h.duplicated(["order_id", "product_id"]).any():
        raise ValueError("Duplicate order-product rows")
    if len(h) != len(history):
        raise ValueError("Unknown history order")
    basket_sizes = h.groupby("order_id").size().rename("basket_size")
    o = orders.join(basket_sizes, on="order_id")
    u = o.groupby("user_id").agg(
        history_orders=("order_number", "max"),
        basket_mean=("basket_size", "mean"),
        basket_std=("basket_size", "std"),
        interval_mean=("days_since_prior_order", "mean"),
        interval_std=("days_since_prior_order", "std"),
    )
    u["historical_repeat_rate"] = h.groupby("user_id").reordered.mean()
    u["distinct_products"] = h.groupby("user_id").product_id.nunique()
    h = h.join(u.history_orders, on="user_id")
    h["age"] = h.history_orders - h.order_number
    h["recent3"] = h.age.lt(3).astype("int8")
    h["recent5"] = h.age.lt(5).astype("int8")
    h["decayed"] = np.power(.8, h.age)
    c = h.groupby(["user_id", "product_id"]).agg(
        purchase_count=("order_id", "size"),
        first_purchase=("order_number", "min"),
        last_purchase=("order_number", "max"),
        recent3_count=("recent3", "sum"),
        recent5_count=("recent5", "sum"),
        decay_score=("decayed", "sum"),
        cart_position_mean=("add_to_cart_order", "mean"),
    ).reset_index().merge(u.reset_index(), on="user_id", validate="many_to_one")
    c["frequency"] = c.purchase_count / c.history_orders
    c["orders_since_purchase"] = c.history_orders - c.last_purchase
    c["in_last_basket"] = c.orders_since_purchase.eq(0).astype("int8")
    c["frequency_since_first"] = c.purchase_count / (c.history_orders-c.first_purchase+1)
    c["single_purchase"] = c.purchase_count.eq(1).astype("int8")
    c["mean_purchase_gap"] = ((c.last_purchase-c.first_purchase)
                              / (c.purchase_count-1).replace(0, np.nan))
    c["gap_ratio"] = c.orders_since_purchase / c.mean_purchase_gap.replace(0, np.nan)
    # Population statistics are fitted exclusively to training users' PRIOR baskets.
    th = h[h.user_id.isin(train_users)]
    g = th.groupby("product_id").agg(global_count=("order_id", "size"),
                                     global_repeat_rate=("reordered", "mean"))
    g["global_support"] = g.global_count / th.order_id.nunique()
    c = c.merge(g, on="product_id", how="left", validate="many_to_one")
    c["unseen_in_training"] = c.global_count.isna().astype("int8")
    c = c.merge(products, on="product_id", validate="many_to_one")
    hd = h[["user_id", "product_id"]].merge(products[["product_id", "department_id"]],
                                            on="product_id", validate="many_to_one")
    shares = pd.crosstab(hd.user_id, hd.department_id, normalize="index")
    shares = shares.reindex(columns=sorted(products.department_id.unique()), fill_value=0)
    shares.columns = [f"dept_share_{int(x)}" for x in shares.columns]
    users = u.join(shares).fillna(0).reset_index()
    affinity = hd.groupby(["user_id", "department_id"]).size().rename("dept_count").reset_index()
    affinity["dept_affinity"] = affinity.dept_count / affinity.groupby("user_id").dept_count.transform("sum")
    c = c.merge(affinity[["user_id", "department_id", "dept_affinity"]],
                on=["user_id", "department_id"], validate="many_to_one")
    c = c.sort_values(["user_id", "product_id"]).reset_index(drop=True)
    return c, users, h


def prepare(raw: Path, cache: Path, sample_size=10000, seed=42):
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "manifest.json").exists():
        raise FileExistsError("Cache already exists. Use a new cache directory for a new protocol.")
    names = ["orders.csv", "products.csv", "aisles.csv", "departments.csv",
             "order_products__prior.csv", "order_products__train.csv"]
    source = {}
    for name in names:
        path = raw / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        source[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    orders = pd.read_csv(raw / "orders.csv")
    products = pd.read_csv(raw / "products.csv")
    terminal = orders[orders.eval_set.eq("train")]
    splits = split_users(terminal.user_id, sample_size, seed)
    selected = orders[orders.user_id.isin(splits.user_id)].copy()
    prior_orders = selected[selected.eval_set.eq("prior")].copy()
    target_orders = selected[selected.eval_set.eq("train")].copy()
    if target_orders.user_id.duplicated().any():
        raise ValueError("More than one target order per user")
    maxhist = prior_orders.groupby("user_id").order_number.max()
    if not np.all(target_orders.order_number.to_numpy() == target_orders.user_id.map(maxhist).to_numpy()+1):
        raise ValueError("Targets must follow all feature history")
    ids = set(prior_orders.order_id)
    chunks = [chunk[chunk.order_id.isin(ids)] for chunk in pd.read_csv(
        raw / "order_products__prior.csv", chunksize=1000000)]
    history = pd.concat(chunks, ignore_index=True)
    if history.order_id.nunique() != len(prior_orders):
        raise ValueError("Some prior baskets have no content")
    train_users = splits.loc[splits.split.eq("train"), "user_id"]
    candidates, users, merged_history = build_candidates(history, prior_orders, products, train_users)
    labels = pd.read_csv(raw / "order_products__train.csv")
    labels = labels.merge(target_orders[["order_id", "user_id"]], on="order_id", validate="many_to_one")
    if labels.user_id.nunique() != sample_size:
        raise ValueError("Missing target labels")
    candidates = candidates.merge(splits, on="user_id", validate="many_to_one")
    users = users.merge(splits, on="user_id", validate="one_to_one")
    candidates.to_parquet(cache / "candidates.parquet", index=False)
    users.to_parquet(cache / "users.parquet", index=False)
    merged_history.to_parquet(cache / "history.parquet", index=False)
    products.to_parquet(cache / "products.parquet", index=False)
    pd.read_csv(raw / "departments.csv").to_parquet(cache / "departments.parquet", index=False)
    splits.to_csv(cache / "splits.csv", index=False)
    # Test labels are separated and not opened by the validation/iteration command.
    for split in ["train", "valid", "test"]:
        subset = labels[labels.user_id.isin(splits.loc[splits.split.eq(split), "user_id"])]
        subset[["user_id", "product_id"]].to_parquet(cache / f"labels_{split}.parquet", index=False)
    manifest = {"protocol_version": 1, "seed": seed, "sample_size": sample_size,
                "candidate_scope": "all historically purchased products; no negative sampling",
                "prediction_time": "immediately after last observed basket; next-order metadata unavailable",
                "source_files": source, "split_counts": splits.split.value_counts().to_dict(),
                "history_baskets": int(history.order_id.nunique()), "history_rows": len(history),
                "candidate_rows": len(candidates), "global_fit_users": len(train_users)}
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
