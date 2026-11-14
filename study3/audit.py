"""Independent raw-data and prediction audit for Study 3.

Does not call the production evaluator, ranker, rule scorer or model trainer.
The audit verifies stored predictions, not a re-fit or model re-selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_metrics(top, repeats):
    """Compute four user metrics from ordered item IDs and raw target sets."""
    rows = []
    for user, group in top.groupby("user_id", sort=False):
        items = group.product_id.to_list()
        truth = repeats[user]
        relevance = [int(item in truth) for item in items]
        hit = sum(relevance)
        count = len(truth)
        length = len(items)
        ideal = sum(1/np.log2(rank+2) for rank in range(min(count, length)))
        actual = sum(value/np.log2(rank+2) for rank, value in enumerate(relevance))
        rows.append({"user_id": user, "repeat_targets": count, "hits": hit,
                     "recommended": length, "recall": hit/count if count else np.nan,
                     "precision": hit/length, "f1": 2*hit/(count+length),
                     "ndcg": actual/ideal if ideal else np.nan})
    return pd.DataFrame(rows).set_index("user_id").sort_index()


def _check_pair_table(actual, expected, key):
    assert not actual.duplicated(key).any(), "Duplicate user-product rows"
    pd.testing.assert_frame_equal(actual[key].sort_values(key).reset_index(drop=True),
                                  expected[key].sort_values(key).reset_index(drop=True),
                                  check_dtype=False)


def verify(cache, run):
    cache, run = Path(cache), Path(run)
    frozen = _json(run/"freeze.json")
    protocol = frozen["protocol"]
    for filename, digest in frozen["sha256"].items():
        assert Path(filename).exists() and _sha(filename) == digest, f"Frozen file changed: {filename}"
    assert _json(run/"confirmation/completed.json")["freeze_sha256"] == _sha(run/"freeze.json")
    assert _json(run/"confirmation/started.json")["freeze_sha256"] == _sha(run/"freeze.json")
    assert frozen["selection"] == _json(run/"selection.json")
    assert frozen["selection"]["selected_before_confirmation"] is True

    # Validate that selections really came from the recorded validation grids.
    quota_grid = pd.read_csv(run/"quota_validation.csv")
    best_quota = quota_grid.sort_values(["recall", "weight"], ascending=[False, True]).iloc[0]
    assert float(best_quota.weight) == frozen["selection"]["quota_weight"]
    tifu_grid = pd.read_csv(run/"tifu_validation.csv")
    best_tifu = tifu_grid.sort_values(["recall", "alpha", "neighbors", "settings_index"],
                                     ascending=[False, False, True, True]).iloc[0]
    for key in ["settings_index", "neighbors", "alpha"]:
        assert best_tifu[key] == frozen["selection"]["tifu"][key], f"TIFU selection mismatch: {key}"

    manifest = _json(cache/"manifest.json")
    raw_dir = Path("archive")
    for name, info in manifest["source"].items():
        assert _sha(raw_dir/name) == info["sha256"], f"Raw source changed: {name}"

    splits = pd.read_csv(cache/"splits.csv")
    assert splits.user_id.nunique() == len(splits)
    expected_counts = {"train": protocol["train_users"], "valid": protocol["validation_users"],
                       "confirm": protocol["confirmation_users"]}
    assert splits.split.value_counts().to_dict() == expected_counts
    previous = pd.concat([pd.read_csv(p)[["user_id"]] for p in protocol["excluded_sources"]])
    assert not previous.user_id.duplicated().any(), "Prior studies unexpectedly overlap"
    assert previous.user_id.nunique() == 43000, "Expected all 43,000 previously used users"
    excluded = pd.read_csv(protocol["excluded_users"])
    assert set(excluded.user_id) == set(previous.user_id)
    assert not splits.user_id.isin(previous.user_id).any(), "New study overlaps previous users"
    confirm_ids = set(splits.loc[splits.split.eq("confirm"), "user_id"])
    train_ids = set(splits.loc[splits.split.eq("train"), "user_id"])

    # Independently inspect each serialized TIFU bank for training-only membership.
    bank_files = sorted(run.glob("tifu_model_*.joblib"))
    assert len(bank_files) == len(protocol["tifu_temporal_grid"])
    for path in bank_files:
        bank = joblib.load(path)
        assert set(bank["train_user_ids"]) == train_ids, "TIFU bank membership mismatch"
        assert len(bank["train_user_ids"]) == len(train_ids)
        assert bank["matrix"].shape[0] == len(train_ids)
        assert not set(bank["train_user_ids"]) & confirm_ids

    orders = pd.read_csv(raw_dir/"orders.csv")
    prior_orders = orders[orders.user_id.isin(confirm_ids) & orders.eval_set.eq("prior")]
    terminal_orders = orders[orders.user_id.isin(confirm_ids) & orders.eval_set.eq("train")]
    assert terminal_orders.user_id.nunique() == len(confirm_ids)
    assert not terminal_orders.user_id.duplicated().any()
    max_history = prior_orders.groupby("user_id").order_number.max()
    np.testing.assert_array_equal(terminal_orders.order_number.to_numpy(),
                                   terminal_orders.user_id.map(max_history).to_numpy()+1)

    key = ["user_id", "product_id"]
    raw_targets = pd.read_csv(raw_dir/"order_products__train.csv").merge(
        terminal_orders[["order_id", "user_id"]], on="order_id", validate="many_to_one")
    saved_labels = pd.read_parquet(cache/"labels_confirm.parquet")
    _check_pair_table(saved_labels, raw_targets, key)

    # Reconstruct the COMPLETE candidate set from raw prior CSV, not feature labels.
    prior_ids = set(prior_orders.order_id)
    chunks = [part.loc[part.order_id.isin(prior_ids), ["order_id", "product_id"]]
              for part in pd.read_csv(raw_dir/"order_products__prior.csv", chunksize=1000000)]
    raw_prior = pd.concat(chunks, ignore_index=True).merge(
        prior_orders[["order_id", "user_id"]], on="order_id", validate="many_to_one")
    assert raw_prior.order_id.nunique() == len(prior_orders)
    raw_candidates = raw_prior[key].drop_duplicates().sort_values(key).reset_index(drop=True)
    features = pd.read_parquet(cache/"features.parquet")
    assert "label" not in features.columns
    candidates = features[features.split.eq("confirm")].copy()
    _check_pair_table(candidates, raw_candidates, key)
    assert set(candidates.user_id) == confirm_ids
    for user, group in candidates.groupby("user_id"):
        assert group.history_orders.eq(max_history[user]).all()

    truth = raw_targets.groupby("user_id").product_id.agg(set).to_dict()
    pools = raw_candidates.groupby("user_id").product_id.agg(set).to_dict()
    repeats = {user: pools[user] & truth[user] for user in confirm_ids}
    target_pairs = set(zip(raw_targets.user_id, raw_targets.product_id))
    covered = candidates[key+["co_narrow_covered"]]
    all_names = ["frequency", "decay"]+protocol["learned_models"]+["quota", "tifu", "tifu_self"]
    learned = set(protocol["learned_models"]+["quota"])
    metrics_checked, exposure, independent_per = [], {}, {}
    for name in all_names:
        model_seeds = protocol["model_seeds"] if name in learned else [42]
        seed_per = []
        for seed in model_seeds:
            dest = run/"confirmation"/name/str(seed)
            scores = pd.read_parquet(dest/"scores.parquet")
            _check_pair_table(scores, raw_candidates, key)
            assert np.isfinite(scores.score).all()
            # Sorting uses only saved scores, raw IDs and the declared tie rule.
            expected_top = scores.sort_values(["user_id", "score", "product_id"],
                                              ascending=[True, False, True]).groupby("user_id", sort=False).head(protocol["primary_k"])
            saved_top = pd.read_parquet(dest/"top10.parquet")
            pd.testing.assert_frame_equal(saved_top[key].reset_index(drop=True),
                                          expected_top[key].reset_index(drop=True), check_dtype=False)
            np.testing.assert_allclose(saved_top.score, expected_top.score, atol=0, rtol=0)
            np.testing.assert_array_equal(saved_top["rank"], saved_top.groupby("user_id").cumcount()+1)
            labels = np.array([int(pair in target_pairs) for pair in zip(saved_top.user_id, saved_top.product_id)])
            np.testing.assert_array_equal(saved_top.label, labels)
            actual = independent_metrics(expected_top, repeats)
            assert set(actual.index) == confirm_ids
            saved_per = pd.read_parquet(dest/"users.parquet").set_index("user_id").sort_index()
            assert not saved_per.index.duplicated().any()
            np.testing.assert_array_equal(actual.index, saved_per.index)
            for field in actual.columns:
                np.testing.assert_allclose(actual[field], saved_per[field], atol=1e-12, rtol=0, equal_nan=True)
            saved_metrics = _json(dest/"metrics.json")
            for metric in ["recall", "precision", "f1", "ndcg"]:
                np.testing.assert_allclose(actual[metric].mean(), saved_metrics[f"{metric}_at_10"], atol=1e-12, rtol=0)
            assert saved_metrics["users"] == len(confirm_ids)
            assert saved_metrics["empty_repeat_targets"] == sum(not x for x in repeats.values())
            if name in {"full", "quota"}:
                exposure[name, seed] = saved_top.merge(covered, on=key, validate="one_to_one").groupby("user_id").agg(
                    covered=("co_narrow_covered", "sum"), length=("product_id", "size"))
            metrics_checked.append({"name": name, "seed": seed, "independent_recall": float(actual.recall.mean())})
            seed_per.append(actual.reset_index())
        independent_per[name] = pd.concat(seed_per).groupby("user_id").mean(numeric_only=True)
        aggregated = pd.read_parquet(run/"confirmation"/name/"users.parquet").set_index("user_id").sort_index()
        for field in independent_per[name].columns:
            np.testing.assert_allclose(independent_per[name][field].sort_index(), aggregated[field], atol=1e-12, rtol=0, equal_nan=True)

    for seed in protocol["model_seeds"]:
        pd.testing.assert_frame_equal(exposure["full", seed].sort_index(), exposure["quota", seed].sort_index(),
                                      check_dtype=False)

    report = {"status": "PASS", "frozen_files": len(frozen["sha256"]),
              "raw_source_files": len(manifest["source"]), "previous_users_excluded": len(previous),
              "previous_user_overlap": 0, "confirmation_users": len(confirm_ids),
              "confirmation_candidate_rows": len(raw_candidates), "confirmation_target_rows": len(raw_targets),
              "empty_repeat_targets": sum(not values for values in repeats.values()),
              "tifu_training_banks_checked": len(bank_files), "prediction_configurations_checked": len(metrics_checked),
              "quota_user_seed_checks": len(confirm_ids)*len(protocol["model_seeds"]),
              "quota_count_mismatches": 0, "independent_metrics": metrics_checked,
              "scope": "raw source hashes, disjoint users, chronology, all historical candidates, raw labels, validation selections, training-only TIFU banks, exact ranking, four per-user metrics, seed aggregation and quota exposures"}
    (run/"audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report
