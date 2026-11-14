"""Posthoc list-length and target-rank diagnostics on frozen Study 4 scores.

No model is fitted and no candidate setting is selected here. Every metric keeps
the original user universe and full-history next-basket repeat denominator.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from basketlab.metrics import attach_labels
from research.analysis import paired_interval
from research.data import save_json
from study4.memory import candidate_mask, evaluate_fixed

KEYS = ['user_id', 'product_id']
RANK_BANDS = ['1-10', '11-20', '21-50', '>50']
PRODUCT_GROUPS = ['recent', 'dormant_repeat', 'dormant_low_frequency']
METRIC_COLUMNS = ['repeat_targets', 'full_candidates', 'candidates', 'covered_targets',
                  'hits', 'recommended', 'recall', 'precision', 'fixed_budget_precision',
                  'f1', 'ndcg', 'coverage', 'oracle_recall', 'candidate_retention']


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _keys(frame):
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError('User/product keys must be unique and nonnull')
    return pd.MultiIndex.from_frame(frame[KEYS])


def pruning_ledger(frame, full_top, retained_top, retained_mask, k):
    """Exact full-to-pruned top-K hit change, retaining empty-target users.

    With identical scores and tie rules, no retained full-top-K item can leave
    the shorter candidate list. Retained items newly promoted from below K can
    replace deleted items; their true hits minus deleted true hits is the gain.
    """
    index = _keys(frame)
    mask = np.asarray(retained_mask)
    if mask.dtype.kind != 'b' or mask.shape != (len(frame),):
        raise ValueError('Retained mask must be a row-aligned boolean vector')
    if not frame.label.isin([0, 1]).all():
        raise ValueError('Full target labels must be binary')
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError('K must be a positive integer')

    def check(top):
        keys = _keys(top)
        rows = index.get_indexer(keys)
        if (rows < 0).any() or top.groupby('user_id').size().gt(k).any():
            raise ValueError('Top list must be drawn from the full frame with at most K items per user')
        labels = frame.label.to_numpy()[rows]
        if 'label' in top and not np.array_equal(labels, top.label.to_numpy()):
            raise ValueError('Top-list labels differ from the full targets')
        result = top[KEYS].copy()
        result['label'] = labels
        return result, keys, rows

    full, full_keys, full_rows = check(full_top)
    kept, kept_keys, kept_rows = check(retained_top)
    if not mask[kept_rows].all():
        raise ValueError('Pruned top list contains excluded candidates')
    deleted = ~full_keys.isin(kept_keys)
    promoted = ~kept_keys.isin(full_keys)
    if mask[full_rows[deleted]].any():
        raise ValueError('A retained full-top item disappeared: scores or tie rules changed')
    per = frame.groupby('user_id').label.sum().rename('repeat_targets').to_frame()
    for name, values in {
        'full_hits': full.groupby('user_id').label.sum(),
        'retained_hits': kept.groupby('user_id').label.sum(),
        'deleted_true_hits': full.loc[deleted].groupby('user_id').label.sum(),
        'promoted_true_hits': kept.loc[promoted].groupby('user_id').label.sum(),
        'deleted_recommendations': full.loc[deleted].groupby('user_id').size(),
        'promoted_recommendations': kept.loc[promoted].groupby('user_id').size(),
    }.items():
        per[name] = values.reindex(per.index, fill_value=0).astype(np.int64)
    per['delta_hits'] = per.retained_hits - per.full_hits
    np.testing.assert_array_equal(per.delta_hits, per.promoted_true_hits - per.deleted_true_hits)
    denom = per.repeat_targets.replace(0, np.nan)
    per['promoted_true_recall'] = per.promoted_true_hits / denom
    per['deleted_true_recall'] = per.deleted_true_hits / denom
    per['delta_recall'] = per.delta_hits / denom
    return per.reset_index()


def target_rank_counts(frame, scores, retained_mask):
    """All true repeat targets, partitioned by eligibility, history, full rank."""
    _keys(frame)
    scores = np.asarray(scores)
    mask = np.asarray(retained_mask)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and row-aligned')
    if mask.shape != scores.shape or mask.dtype.kind != 'b':
        raise ValueError('Mask must be boolean and row-aligned')
    if not frame.label.isin([0, 1]).all():
        raise ValueError('Full target labels must be binary')
    ranked = frame[KEYS + ['label', 'orders_since_purchase', 'purchase_count']].copy()
    ranked['score'] = scores
    ranked['eligibility'] = np.where(mask, 'retained', 'excluded')
    ranked = ranked.sort_values(['user_id', 'score', 'product_id'], ascending=[True, False, True])
    ranked['rank'] = ranked.groupby('user_id').cumcount() + 1
    targets = ranked.loc[ranked.label.eq(1)].copy()
    targets['rank_band'] = pd.cut(targets['rank'], [0, 10, 20, 50, np.inf], labels=RANK_BANDS)
    targets['product_group'] = np.select(
        [targets.orders_since_purchase.lt(5), targets.purchase_count.ge(3)],
        ['recent', 'dormant_repeat'], default='dormant_low_frequency')
    group = ['eligibility', 'product_group', 'rank_band']
    grid = pd.MultiIndex.from_product([['retained', 'excluded'], PRODUCT_GROUPS, RANK_BANDS], names=group)
    counts = targets.groupby(group, observed=True).size().reindex(grid, fill_value=0).rename('targets').reset_index()
    if int(counts.targets.sum()) != int(frame.label.sum()):
        raise AssertionError('Rank partition lost full-history targets')
    return counts


def _standard_metrics(metrics, k):
    return {'recall': metrics[f'recall_at_{k}'], 'ndcg': metrics[f'ndcg_at_{k}'],
            'precision': metrics[f'precision_at_{k}'],
            'fixed_budget_precision': metrics[f'fixed_budget_precision_at_{k}'],
            'users': metrics['users'], 'nonempty_users': metrics['users'] - metrics['empty_repeat_targets'],
            'empty_repeat_targets': metrics['empty_repeat_targets'],
            'mean_returned': metrics['mean_recommendation_length'],
            'mean_candidates': metrics['mean_candidates'], 'coverage': metrics['candidate_coverage'],
            'oracle_recall': metrics[f'oracle_recall_at_{k}']}


def _verify_old_record(frame, metrics, per, top, old, k, seed, policy):
    """Exact numerical cross-checks against completed Study 4 artifacts."""
    if k == 10:
        folder = old / policy / str(seed)
        previous = _read(folder / 'metrics.json')
        for name, value in metrics.items():
            np.testing.assert_allclose(value, previous[name], rtol=0, atol=1e-12, equal_nan=True,
                                       err_msg=f'K10 {policy}/{seed} metric {name}')
        saved = pd.read_parquet(folder / 'users.parquet').set_index('user_id').sort_index()
        actual = per.set_index('user_id').sort_index()
        pd.testing.assert_index_equal(actual.index, saved.index)
        for field in METRIC_COLUMNS:
            np.testing.assert_allclose(actual[field], saved[field], rtol=0, atol=1e-12, equal_nan=True,
                                       err_msg=f'K10 {policy}/{seed} per-user {field}')
        saved_top = pd.read_parquet(folder / 'top10.parquet').sort_values(['user_id', 'rank'])
        actual_top = top.sort_values(['user_id', 'rank'])
        np.testing.assert_array_equal(actual_top[KEYS + ['rank', 'label']], saved_top[KEYS + ['rank', 'label']])
    elif k in [5, 20]:
        previous = pd.read_csv(old / 'k_sensitivity.csv')
        previous = previous.loc[previous.k.eq(k) & previous.policy.eq(policy) & previous.seed.eq(seed)]
        if len(previous) != 1:
            raise AssertionError('Existing K sensitivity record missing or duplicated')
        for name, value in metrics.items():
            np.testing.assert_allclose(value, previous.iloc[0][name], rtol=0, atol=1e-12, equal_nan=True,
                                       err_msg=f'K{k} {policy}/{seed} metric {name}')


def analyze_depth(cache: Path, run: Path, out: Path) -> dict:
    """Read frozen predictions and write posthoc diagnostics under out/depth."""
    cache, run, out = Path(cache), Path(run), Path(out)
    dest = out / 'depth'
    dest.mkdir(parents=True, exist_ok=True)
    protocol = _read('study4_followup/protocol.json')
    ks, seeds = protocol['depth_k'], protocol['seeds']
    old = run / 'confirmation'
    selection = _read(run / 'selection.json')['policy']
    candidates = pd.read_parquet(cache / 'candidates.parquet')
    candidates = candidates.loc[candidates.split.eq('confirm')]
    frame = attach_labels(candidates, pd.read_parquet(cache / 'labels_confirm.parquet'))
    dual = candidate_mask(frame, **selection)
    budgets = pd.Series(dual, index=frame.index).groupby(frame.user_id).sum().astype(int)
    choices = {'full': candidate_mask(frame, 'all'), 'dual': dual,
               'recency_budget': candidate_mask(frame, 'pure_recency_budget', budget=budgets)}
    metrics_rows, user_tables, ledger_tables, rank_tables = [], [], [], []
    checked_old = 0
    for seed in seeds:
        saved = pd.read_parquet(old / 'full' / str(seed) / 'scores.parquet')
        np.testing.assert_array_equal(saved[KEYS].to_numpy(), frame[KEYS].to_numpy())
        scores = saved.score.to_numpy()
        counts = target_rank_counts(frame, scores, dual)
        counts['seed'] = seed
        rank_tables.append(counts)
        for k in ks:
            tops = {}
            for policy, mask in choices.items():
                metrics, per, top = evaluate_fixed(frame, scores, mask, k)
                _verify_old_record(frame, metrics, per, top, old, k, seed, policy)
                checked_old += int(k in [5, 10, 20])
                metrics_rows.append({'k': k, 'policy': policy, 'seed': seed, **_standard_metrics(metrics, k)})
                user_tables.append(per.assign(k=k, policy=policy, seed=seed))
                tops[policy] = top
            ledger_tables.append(pruning_ledger(frame, tops['full'], tops['dual'], dual, k).assign(k=k, seed=seed))
    seed_metrics = pd.DataFrame(metrics_rows)
    seed_metrics.to_csv(dest / 'seed_metrics.csv', index=False)
    curves = seed_metrics.drop(columns='seed').groupby(['k', 'policy'], as_index=False).mean(numeric_only=True)
    curves.to_csv(dest / 'curve_metrics.csv', index=False)
    user_seed = pd.concat(user_tables, ignore_index=True)
    user_seed.to_parquet(dest / 'seed_users.parquet', index=False)
    users = user_seed.drop(columns='seed').groupby(['k', 'policy', 'user_id'], as_index=False).mean(numeric_only=True)
    users.to_parquet(dest / 'users.parquet', index=False)
    contrasts = []
    for k in ks:
        p = users.loc[users.k.eq(k)]
        for comparator in protocol['depth_comparators']:
            stats = paired_interval(p.loc[p.policy.eq('dual')], p.loc[p.policy.eq(comparator)],
                                    n_boot=protocol['depth_bootstrap_replicates'], seed=protocol['bootstrap_seed'],
                                    family=len(ks) * len(protocol['depth_comparators']))
            contrasts.append({'k': k, 'comparator': comparator, **stats})
    pd.DataFrame(contrasts).to_csv(dest / 'contrasts.csv', index=False)
    ledger_seed = pd.concat(ledger_tables, ignore_index=True)
    ledger_seed.to_parquet(dest / 'ledger_seed_users.parquet', index=False)
    ledger = ledger_seed.drop(columns='seed').groupby(['k', 'user_id'], as_index=False).mean(numeric_only=True)
    ledger.to_parquet(dest / 'ledger_users.parquet', index=False)
    ledger_rows = []
    for k, table in ledger.groupby('k'):
        nonempty = table.loc[table.repeat_targets.gt(0)]
        improved = int(nonempty.delta_recall.gt(1e-12).sum())
        worse = int(nonempty.delta_recall.lt(-1e-12).sum())
        unchanged = len(nonempty) - improved - worse
        left = users.loc[users.k.eq(k) & users.policy.eq('dual')].set_index('user_id').recall.sort_index()
        right = users.loc[users.k.eq(k) & users.policy.eq('full')].set_index('user_id').recall.sort_index()
        np.testing.assert_allclose(table.set_index('user_id').delta_recall.sort_index(), left - right,
                                   rtol=0, atol=1e-12, equal_nan=True)
        ledger_rows.append({'k': int(k), 'users': len(table), 'nonempty_users': len(nonempty),
                            'improved_users': improved, 'unchanged_users': unchanged, 'worse_users': worse,
                            'improved_fraction': improved / len(nonempty), 'unchanged_fraction': unchanged / len(nonempty),
                            'worse_fraction': worse / len(nonempty),
                            **{c: float(nonempty[c].mean()) for c in ['promoted_true_hits', 'deleted_true_hits', 'delta_hits',
                                                                    'promoted_true_recall', 'deleted_true_recall', 'delta_recall']}})
    pd.DataFrame(ledger_rows).to_csv(dest / 'rank_ledger.csv', index=False)
    rank_seed = pd.concat(rank_tables, ignore_index=True)
    rank_seed.to_csv(dest / 'target_rank_seed_counts.csv', index=False)
    ranks = rank_seed.drop(columns='seed').groupby(['eligibility', 'product_group', 'rank_band'], as_index=False).mean()
    total = int(frame.label.sum())
    ranks['share_all_repeat_targets'] = ranks.targets / total
    status_totals = ranks.groupby('eligibility').targets.sum()
    ranks['share_within_eligibility'] = ranks.targets / ranks.eligibility.map(status_totals).replace(0, np.nan)
    ranks.to_csv(dest / 'excluded_target_ranks.csv', index=False)
    np.testing.assert_allclose(ranks.targets.sum(), total, rtol=0, atol=1e-12)
    summary = {'posthoc': True, 'model_fits': 0, 'selection_changed': False, 'frozen_selection': selection,
               'users': int(frame.user_id.nunique()), 'nonempty_users': int(frame.groupby('user_id').label.sum().gt(0).sum()),
               'repeat_targets': total, 'excluded_repeat_targets': int(frame.loc[~dual].label.sum()),
               'k_values': ks, 'policies': list(choices), 'seeds': seeds, 'contrasts': len(contrasts),
               'bootstrap_replicates': protocol['depth_bootstrap_replicates'], 'bootstrap_seed': protocol['bootstrap_seed'],
               'simultaneous_interval_family': len(contrasts), 'old_seed_policy_k_records_reproduced': checked_old,
               'ledger_identity_verified_per_user_seed_and_k': True,
               'full_target_rank_partition_verified': True,
               'uncertainty_scope': 'Posthoc conditional paired-user bootstrap of seed-averaged metrics, not new independent confirmation.',
               'interpretation': 'Full historical features and frozen scores remain. Low model rank does not establish low importance or irreducible difficulty.'}
    save_json(dest / 'summary.json', summary)
    return summary
