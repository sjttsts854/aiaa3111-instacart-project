from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def fit_clusters(users, n_clusters=6, seed=42):
    columns = [x for x in users if x.startswith("dept_share_")] + [
        "basket_mean", "interval_mean", "historical_repeat_rate", "history_orders"]
    train = users.split.eq("train").to_numpy()
    scaler = StandardScaler().fit(users.loc[train, columns])
    x = scaler.transform(users[columns])
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(x[train])
    distances = model.transform(x)
    features = pd.DataFrame({"user_id": users.user_id, "cluster": model.predict(x)})
    for j in range(n_clusters):
        features[f"cluster_distance_{j}"] = distances[:, j]
        features[f"cluster_member_{j}"] = features.cluster.eq(j).astype("int8")
    silhouette = silhouette_score(x[train], model.labels_, sample_size=min(2000, train.sum()), random_state=seed)
    profile = users[train].merge(features[["user_id", "cluster"]], on="user_id")
    profiles = profile.groupby("cluster")[columns].mean()
    profiles.insert(0, "users", profile.groupby("cluster").size())
    return features, profiles, {"columns": columns, "scaler": scaler, "model": model}, float(silhouette)


def fit_rules(history, train_users, top_n=500, min_support=.002, min_count=30):
    h = history[history.user_id.isin(train_users)]
    orders = pd.Index(np.sort(h.order_id.unique()))
    top = h.product_id.value_counts().head(top_n).index.sort_values()
    selected = h[h.product_id.isin(top)]
    matrix = coo_matrix((np.ones(len(selected), dtype=bool),
                        (orders.get_indexer(selected.order_id), top.get_indexer(selected.product_id))),
                       shape=(len(orders), len(top))).tocsr()
    # mlxtend requires string names for sparse columns unless integer names start at zero.
    baskets = pd.DataFrame.sparse.from_spmatrix(matrix, columns=top.astype(str))
    support = max(min_support, min_count / len(orders))
    itemsets = fpgrowth(baskets, min_support=support, use_colnames=True, max_len=2)
    if not itemsets.itemsets.map(len).eq(2).any():
        rules = pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])
    else:
        raw = association_rules(itemsets, num_itemsets=len(orders), metric="lift", min_threshold=1.1)
        rules = raw[(raw.antecedents.map(len)==1) & (raw.consequents.map(len)==1)].copy()
        rules["antecedent"] = rules.antecedents.map(lambda x: int(next(iter(x))))
        rules["consequent"] = rules.consequents.map(lambda x: int(next(iter(x))))
        rules = rules[["antecedent", "consequent", "support", "confidence", "lift"]]
    return rules.reset_index(drop=True), {"training_baskets": len(orders), "top_products": len(top),
                                         "min_support": support, "rules": len(rules)}


def rule_features(history, rules):
    last = history[history.age.eq(0)][["user_id", "product_id"]].rename(columns={"product_id": "antecedent"})
    columns = ["user_id", "product_id", "rule_confidence", "rule_lift", "rule_hits"]
    if rules.empty:
        return pd.DataFrame(columns=columns)
    hits = last.merge(rules, on="antecedent")
    features = hits.groupby(["user_id", "consequent"]).agg(
        rule_confidence=("confidence", "max"), rule_lift=("lift", "max"),
        rule_hits=("antecedent", "nunique")).reset_index().rename(columns={"consequent": "product_id"})
    return features


def enrich(cache: Path, seed=42):
    users = pd.read_parquet(cache / "users.parquet")
    history = pd.read_parquet(cache / "history.parquet")
    train_users = users.loc[users.split.eq("train"), "user_id"]
    clusters, profiles, pipeline, silhouette = fit_clusters(users, seed=seed)
    rules, metadata = fit_rules(history, train_users)
    features = rule_features(history, rules)
    c = pd.read_parquet(cache / "candidates.parquet")
    c = c.merge(clusters, on="user_id", validate="many_to_one")
    c = c.merge(features, on=["user_id", "product_id"], how="left", validate="one_to_one")
    for col in ["rule_confidence", "rule_lift", "rule_hits"]:
        c[col] = c[col].fillna(0).astype(float)
    c["rule_covered"] = c.rule_hits.gt(0).astype("int8")
    c.to_parquet(cache / "features.parquet", index=False)
    clusters.to_parquet(cache / "clusters.parquet", index=False)
    profiles.to_csv(cache / "cluster_profiles.csv")
    rules.to_csv(cache / "rules.csv", index=False)
    joblib.dump(pipeline, cache / "clustering.joblib")
    metadata.update({"clusters": 6, "silhouette_train": silhouette,
                     "candidate_rule_coverage": float(c.rule_covered.mean()),
                     "fit_scope": "training users' prior baskets only"})
    (cache / "enrichment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
