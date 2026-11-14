"""Independent Study 4 audit from raw historical/terminal baskets.

No production evaluator, candidate selector, ranker or ledger is imported.
The saved fitted-model scores are audited, not re-fitted or re-selected.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["user_id", "product_id"]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(frame):
    return set(zip(frame.user_id, frame.product_id))


def _check_pairs(actual, expected):
    assert not actual.duplicated(KEYS).any(), "Duplicate user/product key"
    assert _pairs(actual) == _pairs(expected), "User/product universe mismatch"


def policy_pairs(history, policy, budgets=None):
    """Select solely from four explicitly history-derived input columns."""
    columns = KEYS + ["purchase_count", "orders_since_purchase"]
    observed = history[columns].copy()
    kind = policy["kind"]
    if kind == "all":
        return _pairs(observed)
    recent = observed.orders_since_purchase.lt(policy.get("window", 5))
    if kind == "recent":
        return _pairs(observed.loc[recent])
    if kind == "dual":
        result = _pairs(observed.loc[recent])
        work = observed.loc[~recent]
    else:
        result = set()
        work = observed
    if kind == "pure_recency_budget":
        order, ascend = ["orders_since_purchase", "purchase_count", "product_id"], [True, False, True]
    else:
        assert kind in {"dual", "pure_frequency_budget"}
        order, ascend = ["purchase_count", "orders_since_purchase", "product_id"], [False, True, True]
    for user, group in work.groupby("user_id", sort=True):
        limit = int(policy["m"]) if kind == "dual" else int(budgets[user])
        result.update(_pairs(group.sort_values(order, ascending=ascend).head(limit)))
    return result


def independent_evaluate(scores, allowed, repeats, pools, k):
    """Use raw sets and explicit full-user iteration, including empty eligibility."""
    allowed_rows = np.array([pair in allowed for pair in zip(scores.user_id, scores.product_id)])
    ranked = scores.loc[allowed_rows, KEYS + ["score"]].sort_values(
        ["user_id", "score", "product_id"], ascending=[True, False, True])
    top = ranked.groupby("user_id", sort=False).head(k).copy()
    top["rank"] = top.groupby("user_id").cumcount() + 1
    top["label"] = [int(product in repeats[user]) for user, product in zip(top.user_id, top.product_id)]
    lists = top.groupby("user_id").product_id.agg(list).to_dict()
    permitted = ranked.groupby("user_id").product_id.agg(set).to_dict()
    rows = []
    for user in sorted(pools):
        truth, universe = repeats[user], pools[user]
        products = lists.get(user, [])
        eligible = permitted.get(user, set())
        assert eligible <= universe
        hit = len(set(products) & truth)
        count, length = len(truth), len(products)
        covered = len(eligible & truth)
        relevance = [int(product in truth) for product in products]
        ideal = sum(1 / np.log2(position + 2) for position in range(min(k, count)))
        gain = sum(value / np.log2(position + 2) for position, value in enumerate(relevance))
        rows.append({"user_id": user, "repeat_targets": count,
                     "full_candidates": len(universe), "candidates": len(eligible),
                     "covered_targets": covered, "hits": hit, "recommended": length,
                     "recall": hit / count if count else np.nan,
                     "precision": hit / length if length else 0.,
                     "fixed_budget_precision": hit / k,
                     "f1": 2 * hit / (count + length) if count + length else 0.,
                     "ndcg": gain / ideal if ideal else np.nan,
                     "coverage": covered / count if count else np.nan,
                     "oracle_recall": min(k, covered) / count if count else np.nan,
                     "candidate_retention": len(eligible) / len(universe)})
    per = pd.DataFrame(rows).set_index("user_id")
    metrics = {f"{field}_at_{k}": float(per[field].mean()) for field in
               ["recall", "precision", "fixed_budget_precision", "f1", "ndcg", "oracle_recall"]}
    metrics.update(users=len(per), empty_repeat_targets=int(per.repeat_targets.eq(0).sum()),
                   mean_recommendation_length=float(per.recommended.mean()),
                   mean_candidates=float(per.candidates.mean()), candidate_coverage=float(per.coverage.mean()),
                   candidate_retention=float(per.candidate_retention.mean()))
    return metrics, per, top


def _compare_record(path, scores, allowed, repeats, pools, k):
    metrics, actual, expected_top = independent_evaluate(scores, allowed, repeats, pools, k)
    stored_top = pd.read_parquet(path / "top10.parquet")
    for field in KEYS + ["rank", "label"]:
        np.testing.assert_array_equal(stored_top[field], expected_top[field], err_msg=f"{path}: top {field}")
    np.testing.assert_allclose(stored_top.score, expected_top.score, atol=0, rtol=0)
    stored = pd.read_parquet(path / "users.parquet").set_index("user_id").sort_index()
    assert not stored.index.duplicated().any()
    np.testing.assert_array_equal(stored.index, actual.index)
    for field in actual:
        np.testing.assert_allclose(stored[field], actual[field], atol=1e-12, rtol=0,
                                   equal_nan=True, err_msg=f"{path}: user {field}")
    aggregate = _json(path / "metrics.json")
    for field, value in metrics.items():
        np.testing.assert_allclose(aggregate[field], value, atol=1e-12, rtol=0,
                                   equal_nan=True, err_msg=f"{path}: aggregate {field}")
    return actual, expected_top


def independent_ledger(small, large, allowed_small, repeats, pools):
    a = small.groupby("user_id").product_id.agg(set).to_dict()
    b = large.groupby("user_id").product_id.agg(set).to_dict()
    rows = []
    for user in sorted(pools):
        old, new, truth = a.get(user, set()), b.get(user, set()), repeats[user]
        arrivals, departures = new - old, old - new
        assert all((user, product) not in allowed_small for product in arrivals), "Score identity violated"
        introduced = len(arrivals & truth)
        displaced = len(departures & truth)
        delta = len(new & truth) - len(old & truth)
        assert introduced - displaced == delta
        count = len(truth)
        rows.append({"user_id": user, "repeat_targets": count, "hits_small": len(old & truth),
                     "hits_large": len(new & truth), "new_true_hits": introduced,
                     "lost_old_true_hits": displaced, "delta_hits": delta,
                     "new_recommendations": len(arrivals), "displaced_old_recommendations": len(departures),
                     "new_true_recall": introduced / count if count else np.nan,
                     "lost_old_true_recall": displaced / count if count else np.nan,
                     "delta_recall": delta / count if count else np.nan})
    return pd.DataFrame(rows).set_index("user_id")


def _bootstrap_delta(left, right, count, family=1):
    joined = left[["recall"]].join(right[["recall"]], lsuffix="_a", rsuffix="_b", validate="one_to_one")
    values = (joined.recall_a - joined.recall_b).dropna().to_numpy()
    rng = np.random.default_rng(42)
    draws = np.empty(count)
    for start in range(0, count, 100):
        size = min(100, count - start)
        draws[start:start + size] = values[rng.integers(len(values), size=(size, len(values)))].mean(axis=1)
    limits = np.quantile(draws, [.025, .975, .025 / family, 1 - .025 / family])
    return {"users": len(values), "delta": float(values.mean()), "ci95_low": limits[0], "ci95_high": limits[1],
            "sim_low": limits[2], "sim_high": limits[3]}


def _verify_selection(cache, run, frozen):
    protocol = frozen["protocol"]
    frame = pd.read_parquet(cache / "candidates.parquet")
    frame = frame.loc[frame.split.eq("valid")]
    labels = pd.read_parquet(cache / "labels_valid.parquet")
    pools = frame.groupby("user_id").product_id.agg(set).to_dict()
    truth = labels.groupby("user_id").product_id.agg(set).to_dict()
    repeats = {user: products & truth[user] for user, products in pools.items()}
    scores = []
    baseline = []
    for seed in protocol["model_seeds"]:
        path = run / "validation/full" / str(seed)
        values = pd.read_parquet(path / "scores.parquet")
        _check_pairs(values, frame)
        actual, _ = _compare_record(path, values, _pairs(frame), repeats, pools, protocol["primary_k"])
        scores.append(values)
        baseline.append(actual.reset_index())
    base = pd.concat(baseline).groupby("user_id").mean(numeric_only=True)
    stored = pd.read_csv(run / "policy_validation.csv").set_index("old_count")
    assert sorted(stored.index.tolist()) == sorted(protocol["dual_old_counts"])
    grid = []
    for count in protocol["dual_old_counts"]:
        allowed = policy_pairs(frame, {"kind": "dual", "window": protocol["dual_window"], "m": count})
        per = [independent_evaluate(values, allowed, repeats, pools, protocol["primary_k"])[1].reset_index()
               for values in scores]
        actual = pd.concat(per).groupby("user_id").mean(numeric_only=True)
        summary = {"old_count": count, "mean_candidates": float(actual.candidates.mean()),
                   "recall": float(actual.recall.mean()),
                   **_bootstrap_delta(actual, base, protocol["bootstrap_replicates"])}
        summary["eligible"] = bool(summary["ci95_low"] >= -protocol["practical_margin"])
        for field in ["mean_candidates", "recall", "users", "delta", "ci95_low", "ci95_high"]:
            np.testing.assert_allclose(stored.loc[count, field], summary[field], atol=1e-12, rtol=0)
        assert bool(stored.loc[count, "eligible"]) == summary["eligible"]
        grid.append(summary)
    passing = [row for row in grid if row["eligible"]]
    if passing:
        best = sorted(passing, key=lambda row: (row["mean_candidates"], -row["recall"], row["old_count"]))[0]
        choice = {"kind": "dual", "window": protocol["dual_window"], "m": best["old_count"]}
    else:
        choice = {"kind": "all"}
    assert choice == frozen["selection"]["policy"], "Validation-selected policy mismatch"
    return len(grid)


def _raw_confirmation(cache, protocol):
    manifest = _json(cache / "manifest.json")
    raw = Path("archive")
    for filename, info in manifest["source"].items():
        assert _sha(raw / filename) == info["sha256"], f"Raw input changed: {filename}"
    splits = pd.read_csv(cache / "splits.csv")
    assert not splits.user_id.duplicated().any()
    assert splits.split.value_counts().to_dict() == {"train": protocol["train_users"],
            "valid": protocol["validation_users"], "confirm": protocol["confirmation_users"]}
    previous = pd.concat([pd.read_csv(path)[["user_id"]] for path in protocol["excluded_sources"]])
    assert len(previous) == previous.user_id.nunique() == 76000
    assert set(pd.read_csv(protocol["excluded_users"]).user_id) == set(previous.user_id)
    assert not splits.user_id.isin(previous.user_id).any()
    users = set(splits.loc[splits.split.eq("confirm"), "user_id"])
    orders = pd.read_csv(raw / "orders.csv")
    prior_orders = orders.loc[orders.user_id.isin(users) & orders.eval_set.eq("prior")]
    terminal = orders.loc[orders.user_id.isin(users) & orders.eval_set.eq("train")]
    assert set(terminal.user_id) == users and not terminal.user_id.duplicated().any()
    maximum = prior_orders.groupby("user_id").order_number.max()
    np.testing.assert_array_equal(terminal.order_number, terminal.user_id.map(maximum) + 1)
    targets = pd.read_csv(raw / "order_products__train.csv").merge(
        terminal[["user_id", "order_id"]], on="order_id", validate="many_to_one")
    _check_pairs(pd.read_parquet(cache / "labels_confirm.parquet"), targets)
    prior_ids = set(prior_orders.order_id)
    parts = [chunk.loc[chunk.order_id.isin(prior_ids), ["order_id", "product_id"]]
             for chunk in pd.read_csv(raw / "order_products__prior.csv", chunksize=1000000)]
    history = pd.concat(parts, ignore_index=True).merge(
        prior_orders[["user_id", "order_id", "order_number"]], on="order_id", validate="many_to_one")
    assert history.order_id.nunique() == len(prior_ids)
    assert not history.duplicated(["order_id", "product_id"]).any()
    history["age"] = history.user_id.map(maximum) - history.order_number
    history["weight"] = np.power(protocol["decay"], history.age)
    observed = history.groupby(KEYS).agg(purchase_count=("order_id", "size"),
                orders_since_purchase=("age", "min"), decay_score=("weight", "sum")).reset_index()
    observed["history_orders"] = observed.user_id.map(maximum)
    observed["frequency"] = observed.purchase_count / observed.history_orders
    basket_means = history.groupby(["user_id", "order_id"]).size().groupby("user_id").mean()
    observed["basket_mean"] = observed.user_id.map(basket_means)
    index = pd.MultiIndex.from_frame(observed[KEYS])
    for window in [protocol["core_window"]] + protocol["sensitivity_windows"]:
        values = history.loc[history.age.lt(window)].groupby(KEYS).agg(
            frequency=("order_id", "size"), decay=("weight", "sum")).reindex(index, fill_value=0)
        observed[f"frequency_w{window}"] = values.frequency.to_numpy()
        observed[f"decay_w{window}"] = values.decay.to_numpy()
    candidates = pd.read_parquet(cache / "candidates.parquet")
    assert "label" not in candidates
    candidates = candidates.loc[candidates.split.eq("confirm")].sort_values(KEYS)
    _check_pairs(candidates, observed)
    for field in ["purchase_count", "orders_since_purchase", "history_orders", "frequency", "decay_score", "basket_mean"]:
        np.testing.assert_allclose(candidates[field], observed[field], atol=1e-12, rtol=0)
    for window in [3, 5, 10]:
        np.testing.assert_array_equal(candidates[f"recent{window}_count"], observed[f"frequency_w{window}"])
    saved_history = pd.read_parquet(cache / "history.parquet")
    saved_history = saved_history.loc[saved_history.user_id.isin(users)]
    keys = ["user_id", "order_id", "product_id", "order_number", "age"]
    pd.testing.assert_frame_equal(history[keys].sort_values(keys).reset_index(drop=True),
        saved_history[keys].sort_values(keys).reset_index(drop=True), check_dtype=False)
    pools = observed.groupby("user_id").product_id.agg(set).to_dict()
    truth = targets.groupby("user_id").product_id.agg(set).to_dict()
    repeats = {user: pools[user] & truth[user] for user in users}
    return observed, pools, repeats, len(targets), len(manifest["source"])


def verify(cache, run):
    cache, run = Path(cache), Path(run)
    frozen = _json(run / "freeze.json")
    protocol = frozen["protocol"]
    for filename, digest in frozen["sha256"].items():
        assert Path(filename).exists() and _sha(filename) == digest, f"Frozen file changed: {filename}"
    selection = _json(run / "selection.json")
    assert frozen["selection"] == selection and selection["selected_before_confirmation"] is True
    start = _json(run / "confirmation/started.json")
    end = _json(run / "confirmation/completed.json")
    assert start["freeze_sha256"] == end["freeze_sha256"] == _sha(run / "freeze.json")
    assert datetime.fromisoformat(selection["utc"]) <= datetime.fromisoformat(frozen["utc"])
    assert datetime.fromisoformat(frozen["utc"]) <= datetime.fromisoformat(start["utc"]) <= datetime.fromisoformat(end["utc"])
    selection_count = _verify_selection(cache, run, frozen)
    observed, pools, repeats, target_rows, raw_count = _raw_confirmation(cache, protocol)
    masks = {"full": policy_pairs(observed, {"kind": "all"}),
             "recent": policy_pairs(observed, {"kind": "recent", "window": protocol["dual_window"]}),
             "dual": policy_pairs(observed, selection["policy"])}
    budgets = {user: sum((user, product) in masks["dual"] for product in values) for user, values in pools.items()}
    for name, kind in [("frequency_budget", "pure_frequency_budget"), ("recency_budget", "pure_recency_budget")]:
        masks[name] = policy_pairs(observed, {"kind": kind}, budgets)
    records = []
    for name, allowed in masks.items():
        seed_users = []
        for seed in protocol["model_seeds"]:
            score_path = run / "confirmation/full" / str(seed) / "scores.parquet"
            scores = pd.read_parquet(score_path)
            _check_pairs(scores, observed)
            assert np.isfinite(scores.score).all()
            actual, _ = _compare_record(run / "confirmation" / name / str(seed), scores,
                                       allowed, repeats, pools, protocol["primary_k"])
            if name.endswith("_budget"):
                np.testing.assert_array_equal(actual.candidates, pd.Series(budgets).sort_index())
            seed_users.append(actual.reset_index())
            records.append({"name": name, "seed": seed, "independent_recall": float(actual.recall.mean())})
        mean = pd.concat(seed_users).groupby("user_id").mean(numeric_only=True)
        stored = pd.read_parquet(run / "confirmation" / name / "users.parquet").set_index("user_id").sort_index()
        pd.testing.assert_frame_equal(stored, mean, check_dtype=False, check_exact=False, atol=1e-12, rtol=0)
    # Analysis records and expansion ledgers are checked below via an explicit manifest.
    supplementary = _verify_analysis(run, observed, pools, repeats, protocol)
    result = {"status": "PASS", "frozen_files": len(frozen["sha256"]), "raw_source_files": raw_count,
              "previous_users_excluded": 76000, "previous_user_overlap": 0,
              "confirmation_users": len(pools), "confirmation_candidate_rows": len(observed),
              "confirmation_target_rows": target_rows,
              "empty_repeat_targets": sum(not values for values in repeats.values()),
              "validation_policy_variants_checked": selection_count,
              "prediction_configurations_checked": len(records), "independent_metrics": records,
              **supplementary,
              "scope": "Frozen hashes/timestamps; disjoint users; raw chronology, full candidates and labels; history-only masks; independent validation selection; fixed full-target/user metrics including empty eligibility; exact rankings; seed means; matched per-user budgets; factorial and fixed-score ledgers"}
    (run / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return result


def _verify_analysis(run, observed, pools, repeats, protocol):
    """Reconstruct declared cells from the protocol, never trust a partial manifest."""
    dest = run / "confirmation"
    manifest = _json(dest / "analysis_manifest.json")
    expected = {}
    for window in [protocol["core_window"]] + protocol["sensitivity_windows"]:
        for method in protocol["factorial_scoring"]:
            for score_window in [str(window), "all"]:
                column = (f"{method}_w{window}" if score_window != "all" else
                          {"frequency": "purchase_count", "decay": "decay_score"}[method])
                for pool in [str(window), "all"]:
                    path = f"factorial/{method}_s{score_window}_c{pool}"
                    mask = {"kind": "all"} if pool == "all" else {"kind": "recent", "window": window}
                    expected[path] = {"path": path, "score": column, "mask": mask, "k": 10, "seed": None}
    assert len(manifest["cells"]) == len(expected)
    assert {item["path"]: item for item in manifest["cells"]} == expected
    for item in manifest["cells"]:
        scores = observed[KEYS].copy()
        scores["score"] = observed[item["score"]].to_numpy()
        allowed = policy_pairs(observed, item["mask"])
        _compare_record(dest / item["path"], scores, allowed, repeats, pools, item["k"])
    ledgers = {}
    for method in protocol["factorial_scoring"]:
        ledgers[f"ledgers/{method}.parquet"] = (f"factorial/{method}_sall_c5", f"factorial/{method}_sall_call")
    for seed in protocol["model_seeds"]:
        for name in ["full", "dual"]:
            ledgers[f"ledgers/learned_{name}_{seed}.parquet"] = (f"recent/{seed}", f"{name}/{seed}")
    assert len(manifest["ledgers"]) == len(ledgers)
    assert {item["path"] for item in manifest["ledgers"]} == set(ledgers)
    for item in manifest["ledgers"]:
        assert (item["small_path"], item["large_path"]) == ledgers[item["path"]]
        assert item["small_mask"] == {"kind": "recent", "window": protocol["core_window"]}
        assert item["k"] == protocol["primary_k"]
        small = pd.read_parquet(dest / item["small_path"] / "top10.parquet")
        large = pd.read_parquet(dest / item["large_path"] / "top10.parquet")
        allowed = policy_pairs(observed, item["small_mask"])
        actual = independent_ledger(small, large, allowed, repeats, pools)
        stored = pd.read_parquet(dest / item["path"]).set_index("user_id").sort_index()
        np.testing.assert_array_equal(stored.index, actual.index)
        for field in actual:
            np.testing.assert_allclose(stored[field], actual[field], atol=1e-12, rtol=0, equal_nan=True,
                                       err_msg=f"{item['path']}: {field}")
    _verify_primary(dest, observed, protocol)
    _verify_observed_history(dest, observed)
    return {"factorial_configurations_checked": len(expected), "fixed_score_ledgers_checked": len(ledgers),
            "primary_contrasts_independently_bootstrapped": len(protocol["primary_contrasts"]),
            "observed_history_specification_means_checked": 4}


def _verify_primary(dest, observed, protocol):
    """Check estimands, paired users, multiplicity and final noninferiority flags."""
    stored = pd.read_csv(dest / "primary_contrasts.csv").set_index("contrast")
    assert len(stored) == len(protocol["primary_contrasts"])
    long_users = set(observed.loc[observed.history_orders.ge(16), "user_id"])
    expressions = {}

    def recall(path):
        return pd.read_parquet(dest / path / "users.parquet").set_index("user_id").recall.sort_index()

    for method in protocol["factorial_scoring"]:
        aa = recall(f"factorial/{method}_sall_call")
        ra = recall(f"factorial/{method}_s5_call")
        ar = recall(f"factorial/{method}_sall_c5")
        rr = recall(f"factorial/{method}_s5_c5")
        for effect, values in {"information": aa - ra, "expansion": aa - ar,
                               "interaction": aa - ra - ar + rr}.items():
            expressions[f"{method}_{effect}"] = values.loc[values.index.isin(long_users)]
    for control in ["full", "frequency_budget", "recency_budget"]:
        expressions[f"dual_vs_{control}"] = recall("dual") - recall(control)
    assert set(expressions) == set(stored.index) == {item[0] for item in protocol["primary_contrasts"]}
    for contrast, values in expressions.items():
        left = values.rename("recall").to_frame()
        zero = left.copy()
        zero["recall"] = 0.
        actual = _bootstrap_delta(left, zero, protocol["bootstrap_replicates"], len(expressions))
        for field, value in actual.items():
            np.testing.assert_allclose(stored.loc[contrast, field], value, atol=1e-12, rtol=0,
                                       err_msg=f"{contrast}: {field}")
        assert bool(stored.loc[contrast, "positive_confirmed"]) == bool(actual["sim_low"] > 0)
        assert bool(stored.loc[contrast, "noninferiority_confirmed"]) == bool(actual["sim_low"] >= -protocol["practical_margin"])
        assert stored.loc[contrast, "noninferiority_margin"] == protocol["practical_margin"]


def _verify_observed_history(dest, observed):
    """Audit historical-only strata/support/means without importing their producer.

    This additional check does not independently repeat the conditional history
    bootstrap; only the nine primary comparison intervals are re-bootstrapped.
    """
    per = pd.read_parquet(dest / "full/users.parquet").set_index("user_id").sort_index()
    info = observed.groupby("user_id").agg(history_orders=("history_orders", "first"),
                                            basket_mean=("basket_mean", "first"),
                                            candidates=("product_id", "size"))
    details = per[["recall", "repeat_targets"]].join(info, validate="one_to_one")
    details["group"] = np.select([details.history_orders.le(5), details.history_orders.le(15)],
                                  ["3-5", "6-15"], default="16+")
    output = dest / "history_observed"
    raw = pd.read_csv(output / "raw_overall.csv").set_index("history_group")
    for group, values in details.groupby("group"):
        assert raw.loc[group, "users"] == len(values)
        assert raw.loc[group, "nonempty_users"] == int(values.recall.notna().sum())
        for field, value in {"recall": values.recall.mean(), "candidate_mean": values.candidates.mean(),
                             "historical_basket_mean": values.basket_mean.mean()}.items():
            np.testing.assert_allclose(raw.loc[group, field], value, atol=1e-12, rtol=0)
    groups = pd.read_csv(output / "groups.csv").set_index(["specification", "history_group"])
    summaries = pd.read_csv(output / "sensitivity.csv").set_index("specification")
    valid = details.loc[details.repeat_targets.gt(0)].copy()
    valid["candidate_bin"] = np.searchsorted([0, 20, 50, 100, 200, 1000000], valid.candidates, side="left") - 1
    for with_basket, name in [(False, "candidate_only"), (True, "candidate_historical_basket")]:
        valid["basket_bin"] = (np.searchsorted([0, 5, 10, 15, 25, 1000000], valid.basket_mean,
                                                side="left") - 1 if with_basket else 0)
        cells = {}
        for (candidate, basket, group), values in valid.groupby(["candidate_bin", "basket_bin", "group"]):
            cells.setdefault((candidate, basket), {})[group] = values.recall.to_numpy()
        for minimum in [10, 20]:
            specification = f"{name}_min{minimum}"
            support = {key: values for key, values in cells.items()
                       if all(len(values.get(group, [])) >= minimum for group in ["3-5", "6-15", "16+"])}
            row = summaries.loc[specification]
            assert row.eligible_cells == len(support)
            total = sum(sum(len(values) for values in cell.values()) for cell in support.values())
            assert row.retained_users == total
            if not total:
                assert pd.isna(row.standardized_long_minus_short)
                continue
            means, common = {}, {}
            for group in ["3-5", "6-15", "16+"]:
                joined = np.concatenate([cell[group] for cell in support.values()])
                common[group] = joined.mean()
                means[group] = sum(sum(len(values) for values in cell.values()) / total * cell[group].mean()
                                   for cell in support.values())
                target = groups.loc[(specification, group)]
                assert target.retained_users == len(joined)
                np.testing.assert_allclose(target.standardized_recall, means[group], atol=1e-12, rtol=0)
                np.testing.assert_allclose(target.raw_common_support_recall, common[group], atol=1e-12, rtol=0)
            np.testing.assert_allclose(row.standardized_long_minus_short, means["16+"] - means["3-5"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(row.raw_common_support_long_minus_short, common["16+"] - common["3-5"], atol=1e-12, rtol=0)
