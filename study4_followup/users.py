"""Posthoc user changes and overlap using frozen Study 4 predictions only."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.data import save_json
from study4.memory import candidate_mask

SEEDS = [42, 73, 123]
COMPARATORS = ['full', 'recency_budget', 'frequency_budget', 'recent']
TOLERANCE = 1e-12
BOOTSTRAP_SEED = 20261116


def change_summary(values, n_boot=2000, seed=BOOTSTRAP_SEED):
    """Population contributions, never condition contributions on changing users."""
    delta = np.asarray(values, dtype=float)
    total = len(delta)
    delta = delta[np.isfinite(delta)].copy()
    delta[np.abs(delta) <= TOLERANCE] = 0.
    positive, negative = delta > 0, delta < 0
    n = len(delta)
    changed = positive | negative
    result = {
        'users': total, 'nonempty_users': n, 'empty_target_users': total - n,
        'improved_users': int(positive.sum()), 'unchanged_users': int((~changed).sum()),
        'harmed_users': int(negative.sum()),
        'improved_fraction': float(positive.mean()) if n else None,
        'unchanged_fraction': float((~changed).mean()) if n else None,
        'harmed_fraction': float(negative.mean()) if n else None,
        'mean_delta': float(delta.mean()) if n else None,
        'mean_abs_delta': float(np.abs(delta).mean()) if n else None,
        'gain_contribution': float(np.maximum(delta, 0).mean()) if n else None,
        'loss_contribution': float(np.maximum(-delta, 0).mean()) if n else None,
        'changed_mean_delta': float(delta[changed].mean()) if changed.any() else None,
        'improved_mean_gain': float(delta[positive].mean()) if positive.any() else None,
        'harmed_mean_loss': float(-delta[negative].mean()) if negative.any() else None,
        'ci95_low': None, 'ci95_high': None,
    }
    if n and n_boot:
        rng = np.random.default_rng(seed)
        estimates = np.empty(n_boot)
        for start in range(0, n_boot, 100):
            count = min(100, n_boot - start)
            estimates[start:start+count] = delta[rng.integers(n, size=(count, n))].mean(axis=1)
        result['ci95_low'], result['ci95_high'] = map(float, np.quantile(estimates, [.025, .975]))
    return result


def top_overlap(left, right, user_ids):
    """Retain all users; distinguish ordered lists from their unordered sets."""
    ids = pd.Index(user_ids, name='user_id')
    if ids.has_duplicates:
        raise ValueError('User universe must be unique')
    lists = []
    for top in [left, right]:
        if top.duplicated(['user_id', 'product_id']).any():
            raise ValueError('Duplicate recommendation')
        if not top.user_id.isin(ids).all():
            raise ValueError('Recommendations contain unknown users')
        if top.duplicated(['user_id', 'rank']).any():
            raise ValueError('Duplicate rank')
        lists.append(top.sort_values(['user_id', 'rank']).groupby('user_id').product_id.agg(tuple).to_dict())
    rows = []
    for user in ids:
        a, b = lists[0].get(user, ()), lists[1].get(user, ())
        sa, sb = set(a), set(b)
        intersection, union = len(sa & sb), len(sa | sb)
        rows.append({'user_id': user, 'left_count': len(a), 'right_count': len(b),
                     'intersection': intersection, 'union': union,
                     'jaccard': intersection / union if union else 1.,
                     'ordered_identical': a == b, 'unordered_identical': sa == sb})
    return pd.DataFrame(rows)


def candidate_overlap(frame, dual_mask, recency_mask):
    """Observe candidate differences without labels or model outcomes."""
    data = frame[['user_id', 'product_id', 'orders_since_purchase']].copy()
    for mask in [dual_mask, recency_mask]:
        if np.asarray(mask).dtype.kind != 'b' or len(mask) != len(data):
            raise ValueError('Candidate masks must be row-aligned booleans')
    data['dual_count'] = np.asarray(dual_mask).astype(int)
    data['recency_count'] = np.asarray(recency_mask).astype(int)
    data['intersection'] = (dual_mask & recency_mask).astype(int)
    data['union'] = (dual_mask | recency_mask).astype(int)
    old = data.orders_since_purchase.ge(5).to_numpy()
    data['old_intersection'] = (dual_mask & recency_mask & old).astype(int)
    data['old_union'] = ((dual_mask | recency_mask) & old).astype(int)
    data['recent_count'] = (~old).astype(int)
    data['recent_in_both'] = (dual_mask & recency_mask & ~old).astype(int)
    fields = ['dual_count', 'recency_count', 'intersection', 'union', 'old_intersection',
              'old_union', 'recent_count', 'recent_in_both']
    result = data.groupby('user_id')[fields].sum()
    if not result.dual_count.eq(result.recency_count).all():
        raise ValueError('Candidate budgets are not matched user by user')
    result['jaccard'] = (result.intersection / result.union.replace(0, np.nan)).fillna(1.)
    result['old_jaccard'] = result.old_intersection / result.old_union.replace(0, np.nan)
    result['identical'] = result.intersection.eq(result.union)
    result['all_recent_in_both'] = result.recent_count.eq(result.recent_in_both)
    return result.reset_index()


def difference_attributes(frame, dual_mask, recency_mask):
    """Both row-weighted and user-balanced history properties of candidate parts."""
    recent = frame.orders_since_purchase.lt(5).to_numpy()
    groups = {'dual_only': dual_mask & ~recency_mask,
              'recency_only': recency_mask & ~dual_mask,
              'shared_recent': dual_mask & recency_mask & recent,
              'shared_old': dual_mask & recency_mask & ~recent}
    rows = []
    for name, mask in groups.items():
        selected = frame.loc[mask, ['user_id', 'purchase_count', 'orders_since_purchase']]
        means = selected.groupby('user_id')[['purchase_count', 'orders_since_purchase']].mean()
        rows.append({'candidate_part': name, 'rows': len(selected), 'users': len(means),
                     'row_mean_frequency': float(selected.purchase_count.mean()) if len(selected) else None,
                     'row_mean_age': float(selected.orders_since_purchase.mean()) if len(selected) else None,
                     'user_balanced_mean_frequency': float(means.purchase_count.mean()) if len(means) else None,
                     'user_balanced_mean_age': float(means.orders_since_purchase.mean()) if len(means) else None})
    return pd.DataFrame(rows)


def analyze_users(cache: Path, run: Path, out: Path):
    dest = out / 'users'
    dest.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path('study4_followup/protocol.json').read_text(encoding='utf-8'))
    seeds = protocol['seeds']
    bootstrap = protocol['bootstrap_replicates']
    seed = protocol['bootstrap_seed']
    policy = json.loads((run / 'selection.json').read_text(encoding='utf-8'))['policy']
    frame = pd.read_parquet(cache / 'candidates.parquet')
    frame = frame.loc[frame.split.eq('confirm')].copy()
    ids = pd.Index(sorted(frame.user_id.unique()), name='user_id')
    tables = {}
    for name in ['dual'] + COMPARATORS:
        rows = []
        for model_seed in seeds:
            per = pd.read_parquet(run / 'confirmation' / name / str(model_seed) / 'users.parquet').set_index('user_id').sort_index()
            if not per.index.equals(ids):
                raise ValueError('Saved predictions changed the user universe')
            rows.append(per.assign(seed=model_seed))
        tables[name] = pd.concat(rows)
    reference = tables['full'].loc[tables['full'].seed.eq(seeds[0]), 'repeat_targets']
    for table in tables.values():
        for model_seed in seeds:
            part = table[table.seed.eq(model_seed)]
            if not part.repeat_targets.equals(reference):
                raise ValueError('Target denominator changed between seeds or policies')
            expected = part.hits / part.repeat_targets.replace(0, np.nan)
            np.testing.assert_allclose(expected, part.recall, rtol=0, atol=1e-12, equal_nan=True)
    dual_mask = candidate_mask(frame, **policy)
    budgets = pd.Series(dual_mask, index=frame.index).groupby(frame.user_id).sum().astype(int)
    recency_mask = candidate_mask(frame, 'pure_recency_budget', budget=budgets)
    features = frame.assign(old=frame.orders_since_purchase.ge(5)).groupby('user_id').agg(
        history_orders=('history_orders', 'first'), full_candidates=('product_id', 'size'), old_candidates=('old', 'sum'))
    features['old_catalog_share'] = features.old_candidates / features.full_candidates
    features['dual_candidates'] = budgets
    features['candidate_retention'] = features.dual_candidates / features.full_candidates
    features['history_group'] = pd.cut(features.history_orders, [2, 5, 15, np.inf], labels=protocol['history_groups'])
    features['candidate_group'] = pd.cut(features.full_candidates, protocol['candidate_bins'], labels=['1-20', '21-50', '51-100', '101+'])
    features['old_share_group'] = pd.cut(features.old_catalog_share, protocol['old_catalog_share_bins'], include_lowest=True,
                                          labels=['0-25%', '>25-50%', '>50-75%', '>75%'])
    features['repeat_targets'] = reference
    for name, table in tables.items():
        features[f'recall_{name}'] = table.groupby(level='user_id').recall.mean()
    rows, seed_rows = [], []
    for comparator in COMPARATORS:
        col = f'delta_{comparator}'
        features[col] = features.recall_dual - features[f'recall_{comparator}']
        values = []
        for model_seed in seeds:
            value = tables['dual'].loc[tables['dual'].seed.eq(model_seed), 'recall'] - tables[comparator].loc[tables[comparator].seed.eq(model_seed), 'recall']
            values.append(value)
            seed_rows.append({'comparator': comparator, 'seed': model_seed, **change_summary(value, n_boot=0)})
        changes = pd.concat(values, axis=1).abs().gt(TOLERANCE)
        # NaN is not "unchanged": expose empty targets separately in both flags.
        features[f'any_seed_change_{comparator}'] = changes.any(axis=1).where(reference.gt(0)).astype('boolean')
        features[f'all_seed_unchanged_{comparator}'] = (~changes.any(axis=1)).where(reference.gt(0)).astype('boolean')
        rows.append({'comparator': comparator, **change_summary(features[col], bootstrap, seed),
                     'any_seed_change_users': int(features[f'any_seed_change_{comparator}'].sum()),
                     'all_seed_unchanged_users': int(features[f'all_seed_unchanged_{comparator}'].sum())})
    changes = pd.DataFrame(rows)
    changes.to_csv(dest / 'change_summary.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(dest / 'seed_change_summary.csv', index=False)
    groups = []
    for dimension in ['history_group', 'candidate_group', 'old_share_group']:
        for group in features[dimension].cat.categories:
            part = features.loc[features[dimension].eq(group)]
            for comparator in COMPARATORS:
                groups.append({'dimension': dimension, 'group': str(group), 'comparator': comparator,
                               'mean_candidates': float(part.full_candidates.mean()) if len(part) else None,
                               'mean_dual_candidates': float(part.dual_candidates.mean()) if len(part) else None,
                               'mean_candidate_retention': float(part.candidate_retention.mean()) if len(part) else None,
                               **change_summary(part[f'delta_{comparator}'], bootstrap, seed)})
    pd.DataFrame(groups).to_csv(dest / 'subgroups.csv', index=False)
    features.reset_index().to_parquet(dest / 'users.parquet', index=False)
    candidate = candidate_overlap(frame, dual_mask, recency_mask)
    candidate.to_csv(dest / 'candidate_overlap.csv', index=False)
    difference_attributes(frame, dual_mask, recency_mask).to_csv(dest / 'candidate_difference_attributes.csv', index=False)
    top_rows = []
    for model_seed in seeds:
        dual_top = pd.read_parquet(run / 'confirmation/dual' / str(model_seed) / 'top10.parquet')
        for comparator in COMPARATORS:
            control_top = pd.read_parquet(run / 'confirmation' / comparator / str(model_seed) / 'top10.parquet')
            overlap = top_overlap(dual_top, control_top, ids)
            top_rows.append(overlap.assign(comparator=comparator, seed=model_seed))
    top_seed = pd.concat(top_rows, ignore_index=True)
    top_seed.to_csv(dest / 'top_overlap_seed.csv', index=False)
    top_users = top_seed.groupby(['comparator', 'user_id']).mean(numeric_only=True).drop(columns='seed')
    top_users['all_seeds_ordered_identical'] = top_users.ordered_identical.eq(1.)
    top_users['all_seeds_unordered_identical'] = top_users.unordered_identical.eq(1.)
    top_users.reset_index().to_csv(dest / 'top_overlap_users.csv', index=False)
    summaries = []
    for population, chosen_ids in [('all', ids), ('nonempty', ids[reference.gt(0)])]:
        c = candidate[candidate.user_id.isin(chosen_ids)]
        for comparator in COMPARATORS:
            t = top_users.loc[comparator].loc[chosen_ids]
            row = {'population': population, 'comparator': comparator, 'users': len(chosen_ids),
                   'top_ordered_identical_seedmean': float(t.ordered_identical.mean()),
                   'top_unordered_identical_seedmean': float(t.unordered_identical.mean()),
                   'top_all_seeds_ordered_identical_fraction': float(t.all_seeds_ordered_identical.mean()),
                   'top_all_seeds_unordered_identical_fraction': float(t.all_seeds_unordered_identical.mean()),
                   'top_mean_jaccard': float(t.jaccard.mean()), 'top_mean_intersection': float(t.intersection.mean())}
            if comparator == 'recency_budget':
                row.update(candidate_mean_jaccard=float(c.jaccard.mean()), candidate_identical_fraction=float(c.identical.mean()),
                           candidate_old_mean_jaccard=float(c.old_jaccard.mean()), candidate_old_union_nonempty_users=int(c.old_union.gt(0).sum()),
                           candidate_all_recent_in_both_fraction=float(c.all_recent_in_both.mean()),
                           candidate_total_intersection=int(c.intersection.sum()), candidate_total_union=int(c.union.sum()))
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(dest / 'overlap_summary.csv', index=False)
    summary = {'posthoc': True, 'users': len(features), 'nonempty_users': int(reference.gt(0).sum()),
               'empty_target_users': int(reference.eq(0).sum()), 'seed_aggregation': 'Mean of per-user metrics across fitted models, not prediction ensemble',
               'change_tolerance': TOLERANCE, 'bootstrap_replicates': bootstrap, 'bootstrap_seed': seed,
               'all_candidate_budgets_match': bool(candidate.dual_count.eq(candidate.recency_count).all()),
               'all_recent_products_in_both_candidate_pools': bool(candidate.all_recent_in_both.all()),
               'comparators': COMPARATORS, 'change_summary': rows,
               'subgroup_intervals': 'Exploratory marginal 95%, conditional on frozen models and observed groups; no confirmatory subgroup selection',
               'overlap_scope': 'Lists evaluated per seed then averaged within user; all-seed identity also reported; empty/empty Jaccard=1; old-only candidate Jaccard is undefined for empty old union',
               'outcome_defined_groups': 'Improved/harmed/changed labels are post-outcome diagnostics, never prediction-time targeting rules'}
    save_json(dest / 'summary.json', summary)
    return summary
