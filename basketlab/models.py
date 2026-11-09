from __future__ import annotations

import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_FEATURES = ["purchase_count", "first_purchase", "last_purchase", "recent3_count", "recent5_count",
                 "decay_score", "cart_position_mean", "history_orders", "basket_mean", "basket_std",
                 "interval_mean", "interval_std", "historical_repeat_rate", "distinct_products",
                 "frequency", "orders_since_purchase", "in_last_basket", "frequency_since_first",
                 "single_purchase", "mean_purchase_gap", "gap_ratio", "global_count",
                 "global_repeat_rate", "global_support", "unseen_in_training", "dept_affinity",
                 "aisle_id", "department_id"]
RULE_FEATURES = ["rule_confidence", "rule_lift", "rule_hits", "rule_covered"]


def feature_columns(frame, config):
    columns = BASE_FEATURES.copy()
    if config.get("clusters"):
        columns += sorted(x for x in frame if x.startswith(("cluster_distance_", "cluster_member_")))
    if config.get("rules"):
        columns += RULE_FEATURES
    return columns


def fit_predict(train, evaluation, config):
    start = time.perf_counter()
    family = config["family"]
    model, columns = None, []
    if family == "frequency":
        scores = evaluation.frequency.to_numpy()
    elif family == "recent":
        # Last-basket-first, then historical frequency; product ID breaks exact ties.
        scores = evaluation.in_last_basket.to_numpy()*2 + evaluation.frequency.to_numpy()
    elif family == "decay":
        scores = evaluation.decay_score.to_numpy()
    else:
        columns = feature_columns(train, config)
        x = train[columns].replace([np.inf, -np.inf], np.nan)
        xe = evaluation[columns].replace([np.inf, -np.inf], np.nan)
        if family == "logistic":
            cats = ["aisle_id", "department_id"]
            nums = [x for x in columns if x not in cats]
            preprocessing = ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), nums),
                ("category", OneHotEncoder(handle_unknown="ignore"), cats)])
            model = make_pipeline(preprocessing, LogisticRegression(C=.5, max_iter=500, solver="lbfgs"))
            model.fit(x, train.label)
        elif family == "lightgbm":
            params = dict(n_estimators=180, learning_rate=.05, num_leaves=31, min_child_samples=60,
                          colsample_bytree=1., reg_lambda=1., n_jobs=4, random_state=42,
                          deterministic=True, force_col_wise=True, verbosity=-1)
            params.update(config.get("params", {}))
            model = LGBMClassifier(**params)
            model.fit(x, train.label, categorical_feature=["aisle_id", "department_id"])
        else:
            raise ValueError(f"Unknown model family: {family}")
        scores = model.predict_proba(xe)[:,1]
    elapsed = time.perf_counter()-start
    return np.asarray(scores), model, columns, elapsed


def predict_saved(frame, artifact):
    config = artifact["config"]
    if config["family"] in {"frequency", "recent", "decay"}:
        return fit_predict(None, frame, config)[0]
    x = frame[artifact["columns"]].replace([np.inf, -np.inf], np.nan)
    return artifact["model"].predict_proba(x)[:, 1]
