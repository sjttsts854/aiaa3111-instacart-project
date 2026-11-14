"""Independent consistency audit of posthoc diagnostics, without refitting.

The original study already audited raw baskets. This layer checks the fixed
source artifacts, then independently reconstructs ranks, sets and accounting.
It never imports the followup or production evaluation/selection functions.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['user_id', 'product_id']


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes(expected):
    for path, digest in expected.items():
        assert _sha(path) == digest, f'Changed protected input: {path}'


def history_masks(frame, policy):
    """Rebuild the frozen choice and matched recency mask from historical rows."""
    assert not frame.duplicated(KEYS).any()
    assert policy['kind'] in {'dual', 'all'}
    work = frame[KEYS + ['purchase_count', 'orders_since_purchase']].reset_index(drop=True)
    work['_position'] = np.arange(len(work))
    all_rows = np.ones(len(work), dtype=bool)
    if policy['kind'] == 'all':
        chosen = all_rows.copy()
    else:
        chosen = work.orders_since_purchase.to_numpy() < policy['window']
        old = work.loc[~chosen].sort_values(
            ['user_id', 'purchase_count', 'orders_since_purchase', 'product_id'],
            ascending=[True, False, True, True])
        chosen[old.groupby('user_id', sort=False).head(policy['m'])._position] = True
    budgets = work.loc[chosen].groupby('user_id').size()
    recent_order = work.sort_values(['user_id', 'orders_since_purchase', 'purchase_count', 'product_id'],
                                   ascending=[True, True, False, True])
    rank = recent_order.groupby('user_id').cumcount()
    retained = rank < recent_order.user_id.map(budgets).fillna(0).to_numpy()
    recency = np.zeros(len(work), dtype=bool)
    recency[recent_order.loc[retained, '_position']] = True
    return {'full': all_rows, 'dual': chosen, 'recency_budget': recency}


def ranked_rows(frame, scores, eligible):
    """Exact ordering under the frozen product-ID tie rule."""
    assert len(frame) == len(scores) == len(eligible)
    assert not frame.duplicated(KEYS).any()
    ranked = frame.loc[np.asarray(eligible), KEYS + ['label']].copy()
    ranked['score'] = np.asarray(scores)[eligible]
    ranked = ranked.sort_values(['user_id', 'score', 'product_id'], ascending=[True, False, True])
    ranked['rank'] = ranked.groupby('user_id').cumcount() + 1
    return ranked


def per_at_k(frame, ranked, k):
    truth = frame.groupby('user_id').label.sum().rename('repeat_targets')
    top = ranked.loc[ranked['rank'].le(k)]
    result = truth.to_frame()
    result['hits'] = top.groupby('user_id').label.sum().reindex(result.index, fill_value=0)
    result['recommended'] = top.groupby('user_id').size().reindex(result.index, fill_value=0)
    result['recall'] = result.hits / result.repeat_targets.replace(0, np.nan)
    result['candidates'] = ranked.groupby('user_id').size().reindex(result.index, fill_value=0)
    result['covered_targets'] = ranked.groupby('user_id').label.sum().reindex(result.index, fill_value=0)
    result['coverage'] = result.covered_targets / result.repeat_targets.replace(0, np.nan)
    result['full_candidates'] = frame.groupby('user_id').size()
    result['precision'] = (result.hits / result.recommended.replace(0, np.nan)).fillna(0.)
    result['fixed_budget_precision'] = result.hits / k
    result['f1'] = (2 * result.hits / (result.repeat_targets + result.recommended).replace(0, np.nan)).fillna(0.)
    gains = top.assign(gain=top.label / np.log2(top['rank'] + 1)).groupby('user_id').gain.sum().reindex(result.index, fill_value=0)
    discounts = np.r_[0., np.cumsum(1 / np.log2(np.arange(2, k + 2)))]
    ideal = discounts[np.minimum(k, result.repeat_targets).to_numpy(dtype=int)]
    result['ndcg'] = gains / pd.Series(ideal, index=result.index).replace(0, np.nan)
    result['oracle_recall'] = np.minimum(k, result.covered_targets) / result.repeat_targets.replace(0, np.nan)
    result['candidate_retention'] = result.candidates / result.full_candidates
    return result


def sign_summary(values):
    """Signed population contributions preserve exactly the original denominator."""
    values = pd.Series(values).dropna().to_numpy()
    positive = values > 1e-12
    negative = values < -1e-12
    return {'nonempty_users': len(values), 'improved_users': int(positive.sum()),
            'worse_users': int(negative.sum()), 'unchanged_users': int((~positive & ~negative).sum()),
            'mean_delta': float(values.mean()), 'mean_absolute_delta': float(np.abs(values).mean()),
            'gain_contribution': float(np.where(positive, values, 0).mean()),
            'loss_contribution': float(np.where(negative, values, 0).mean())}


def overlap_rows(left, right, users):
    """Ordered list equality and set overlap are different observed quantities."""
    a = left.groupby('user_id').product_id.agg(list).to_dict()
    b = right.groupby('user_id').product_id.agg(list).to_dict()
    rows = []
    for user in users:
        la, lb = a.get(user, []), b.get(user, [])
        sa, sb = set(la), set(lb)
        union = sa | sb
        rows.append({'user_id': user, 'left_count': len(la), 'right_count': len(lb),
                     'intersection': len(sa & sb), 'union': len(union),
                     'jaccard': len(sa & sb) / len(union) if union else 1.,
                     'ordered_equal': la == lb, 'unordered_equal': sa == sb})
    return pd.DataFrame(rows).set_index('user_id')


def pruning_ledger(frame, full_rank, dual_rank, k):
    """Compare lists via sets; never infer relevance from a pruned denominator."""
    truth = frame.loc[frame.label.eq(1)].groupby('user_id').product_id.agg(set).to_dict()
    a = full_rank.loc[full_rank['rank'].le(k)].groupby('user_id').product_id.agg(set).to_dict()
    b = dual_rank.loc[dual_rank['rank'].le(k)].groupby('user_id').product_id.agg(set).to_dict()
    allowed = dual_rank.groupby('user_id').product_id.agg(set).to_dict()
    rows = []
    for user in sorted(frame.user_id.unique()):
        full, dual, targets = a.get(user, set()), b.get(user, set()), truth.get(user, set())
        removed, promoted = full - dual, dual - full
        assert not removed & allowed.get(user, set()), 'Retained full top-K item was displaced under pruning'
        lost, gained = len(removed & targets), len(promoted & targets)
        delta = len(dual & targets) - len(full & targets)
        assert gained - lost == delta
        denominator = len(targets)
        rows.append({'user_id': user, 'repeat_targets': denominator, 'removed_true_hits': lost,
                     'promoted_true_hits': gained, 'delta_hits': delta,
                     'removed_true_recall': lost / denominator if denominator else np.nan,
                     'promoted_true_recall': gained / denominator if denominator else np.nan,
                     'delta_recall': delta / denominator if denominator else np.nan})
    return pd.DataFrame(rows).set_index('user_id')


def _near(actual, expected, message=''):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True, err_msg=message)


def verify(cache, run, out):
    """Verify the finished posthoc run using frozen source artifacts."""
    cache, run, out = Path(cache), Path(run), Path(out)
    manifest = _read(out / 'inputs.json')
    completed = _read(out / 'completed.json')
    assert manifest['posthoc'] is True and completed['posthoc'] is True
    assert completed['input_manifest_sha256'] == _sha(out / 'inputs.json')
    verify_hashes(manifest['sha256'])
    frozen = _read(run / 'freeze.json')
    verify_hashes(frozen['sha256'])
    assert _read(run / 'audit.json')['status'] == 'PASS'
    source_completed = _read(run / 'confirmation/completed.json')
    assert manifest['source_confirmation_completed'] == source_completed
    assert datetime.fromisoformat(source_completed['utc']) < datetime.fromisoformat(manifest['utc'])
    assert datetime.fromisoformat(manifest['utc']) <= datetime.fromisoformat(completed['utc'])
    protocol = manifest['protocol']
    assert protocol == _read('study4_followup/protocol.json')
    assert protocol['depth_k'] == [5, 10, 15, 20, 30, 50]
    assert protocol['seeds'] == [42, 73, 123]
    assert protocol['depth_comparators'] == ['full', 'recency_budget']
    frame = pd.read_parquet(cache / 'candidates.parquet')
    frame = frame.loc[frame.split.eq('confirm')].reset_index(drop=True)
    labels = pd.read_parquet(cache / 'labels_confirm.parquet').assign(label=1)
    assert not labels.duplicated(KEYS).any()
    frame = frame.merge(labels, on=KEYS, how='left', validate='one_to_one')
    frame['label'] = frame.label.fillna(0).astype(int)
    users = sorted(frame.user_id.unique())
    assert len(users) == 8000
    assert frame.groupby('user_id').label.sum().eq(0).sum() == 528
    policy = frozen['selection']['policy']
    assert _read(run / 'selection.json')['policy'] == policy
    masks = history_masks(frame, policy)
    details = {}
    details.update(_verify_user_outputs(frame, run, out, masks, protocol))
    details.update(_verify_depth_outputs(frame, run, out, masks, protocol))
    result = {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(), 'posthoc': True,
              'source_input_files_unchanged': len(manifest['sha256']),
              'original_frozen_files_unchanged': len(frozen['sha256']),
              'confirmation_users': len(users), 'nonempty_repeat_target_users': 7472,
              'input_manifest_sha256': _sha(out / 'inputs.json'), **details,
              'scope': 'Saved fixed predictions and prior audited data; input hashes and posthoc timestamps; full-user/full-target denominators; individual signed changes; candidate and ordered/unordered top-K overlap; independent ranks, depth metrics and deletion/promotion accounting. No new raw-source audit, fitting or selection.'}
    (out / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return result


def _verify_user_outputs(frame, run, out, masks, protocol):
    dest = out / 'users'; old = run / 'confirmation'
    actual = pd.read_parquet(dest / 'users.parquet').set_index('user_id').sort_index()
    reference = frame.groupby('user_id').agg(history_orders=('history_orders', 'first'), full_candidates=('product_id', 'size'), repeat_targets=('label', 'sum'))
    reference['old_candidates'] = frame.loc[frame.orders_since_purchase.ge(5)].groupby('user_id').size().reindex(reference.index, fill_value=0)
    reference['old_catalog_share'] = reference.old_candidates / reference.full_candidates
    reference['dual_candidates'] = frame.loc[masks['dual']].groupby('user_id').size().reindex(reference.index, fill_value=0)
    reference['candidate_retention'] = reference.dual_candidates / reference.full_candidates
    _compare_frame(actual, reference)
    reference['history_group'] = pd.cut(reference.history_orders, [2, 5, 15, np.inf], labels=['3-5', '6-15', '16+'])
    reference['candidate_group'] = pd.cut(reference.full_candidates, [0, 20, 50, 100, 1e6], labels=['1-20', '21-50', '51-100', '101+'])
    reference['old_share_group'] = pd.cut(reference.old_catalog_share, [0, .25, .5, .75, 1.01], include_lowest=True, labels=['0-25%', '>25-50%', '>50-75%', '>75%'])
    for field in ['history_group', 'candidate_group', 'old_share_group']:
        np.testing.assert_array_equal(actual[field].astype(str), reference[field].astype(str))
    tables = {}; comparators = protocol['user_comparators']; seeds = protocol['seeds']
    for name in ['dual'] + comparators:
        columns = []
        for seed in seeds:
            table = pd.read_parquet(old / name / str(seed) / 'users.parquet').set_index('user_id').sort_index()
            _near(table.repeat_targets, reference.repeat_targets)
            _near(table.recall, table.hits / reference.repeat_targets.replace(0, np.nan))
            columns.append(table.recall.rename(seed))
        tables[name] = pd.concat(columns, axis=1)
        reference[f'recall_{name}'] = tables[name].mean(axis=1)
        _near(actual[f'recall_{name}'], reference[f'recall_{name}'])
    changes = pd.read_csv(dest / 'change_summary.csv').set_index('comparator')
    seed_changes = pd.read_csv(dest / 'seed_change_summary.csv').set_index(['comparator', 'seed'])
    for name in comparators:
        value = reference.recall_dual - reference[f'recall_{name}']
        reference[f'delta_{name}'] = value
        _near(actual[f'delta_{name}'], value)
        _check_change(changes.loc[name], value, protocol['bootstrap_replicates'], protocol['bootstrap_seed'])
        different = (tables['dual'] - tables[name]).abs().gt(1e-12).any(axis=1)
        nonempty = reference.repeat_targets.gt(0)
        for field, flag in [(f'any_seed_change_{name}', different), (f'all_seed_unchanged_{name}', ~different)]:
            expected = flag.where(nonempty).astype('boolean')
            pd.testing.assert_series_equal(actual[field], expected, check_names=False, check_dtype=False)
        assert changes.loc[name, 'any_seed_change_users'] == int((different & nonempty).sum())
        assert changes.loc[name, 'all_seed_unchanged_users'] == int((~different & nonempty).sum())
        for seed in seeds:
            _check_change(seed_changes.loc[(name, seed)], tables['dual'][seed] - tables[name][seed])
    groups = pd.read_csv(dest / 'subgroups.csv')
    assert len(groups) == 44
    for _, row in groups.iterrows():
        values = reference.loc[reference[row.dimension].astype(str).eq(row.group)]
        _check_change(row, values[f'delta_{row.comparator}'])
        for field, column in [('mean_candidates', 'full_candidates'), ('mean_dual_candidates', 'dual_candidates'), ('mean_candidate_retention', 'candidate_retention')]:
            _near(row[field], values[column].mean())
    candidate = overlap_rows(frame.loc[masks['dual']], frame.loc[masks['recency_budget']], reference.index)
    saved_candidate = pd.read_csv(dest / 'candidate_overlap.csv').set_index('user_id').sort_index()
    for field, column in [('dual_count','left_count'),('recency_count','right_count'),('intersection','intersection'),('union','union'),('jaccard','jaccard'),('identical','unordered_equal')]:
        _near(saved_candidate[field], candidate[column])
    assert candidate.left_count.eq(candidate.right_count).all()
    recent = frame.orders_since_purchase.lt(5).to_numpy()
    older = overlap_rows(frame.loc[masks['dual'] & ~recent], frame.loc[masks['recency_budget'] & ~recent], reference.index)
    _near(saved_candidate.old_intersection, older.intersection)
    _near(saved_candidate.old_union, older.union)
    _near(saved_candidate.old_jaccard, older.intersection / older.union.replace(0, np.nan))
    recent_count = frame.loc[recent].groupby('user_id').size().reindex(reference.index, fill_value=0)
    recent_both = frame.loc[recent & masks['dual'] & masks['recency_budget']].groupby('user_id').size().reindex(reference.index, fill_value=0)
    _near(saved_candidate.recent_count, recent_count); _near(saved_candidate.recent_in_both, recent_both)
    _near(saved_candidate.all_recent_in_both, recent_count.eq(recent_both))
    expected_top = []
    for seed in seeds:
        dual = pd.read_parquet(old / 'dual' / str(seed) / 'top10.parquet').sort_values(['user_id', 'rank'])
        for name in comparators:
            other = pd.read_parquet(old / name / str(seed) / 'top10.parquet').sort_values(['user_id', 'rank'])
            values = overlap_rows(dual, other, reference.index).rename(columns={'ordered_equal':'ordered_identical','unordered_equal':'unordered_identical'})
            expected_top.append(values.reset_index().assign(seed=seed, comparator=name))
    top = pd.concat(expected_top, ignore_index=True)
    _compare_frame(pd.read_csv(dest / 'top_overlap_seed.csv').set_index(['comparator','seed','user_id']), top.set_index(['comparator','seed','user_id']))
    top_mean = top.drop(columns='seed').groupby(['comparator','user_id']).mean(numeric_only=True)
    top_mean['all_seeds_ordered_identical'] = top_mean.ordered_identical.eq(1)
    top_mean['all_seeds_unordered_identical'] = top_mean.unordered_identical.eq(1)
    _compare_frame(pd.read_csv(dest / 'top_overlap_users.csv').set_index(['comparator','user_id']), top_mean)
    summary = pd.read_csv(dest / 'overlap_summary.csv')
    for _, row in summary.iterrows():
        ids = reference.index if row.population == 'all' else reference.index[reference.repeat_targets.gt(0)]
        t = top_mean.loc[row.comparator].loc[ids]; c = saved_candidate.loc[ids]
        assert row.users == len(ids)
        for field, column in [('top_ordered_identical_seedmean','ordered_identical'),('top_unordered_identical_seedmean','unordered_identical'),('top_all_seeds_ordered_identical_fraction','all_seeds_ordered_identical'),('top_all_seeds_unordered_identical_fraction','all_seeds_unordered_identical'),('top_mean_jaccard','jaccard'),('top_mean_intersection','intersection')]:
            _near(row[field], t[column].mean())
        if row.comparator == 'recency_budget':
            for field, column in [('candidate_mean_jaccard','jaccard'),('candidate_identical_fraction','identical'),('candidate_old_mean_jaccard','old_jaccard'),('candidate_all_recent_in_both_fraction','all_recent_in_both')]:
                _near(row[field], c[column].mean())
            assert row.candidate_old_union_nonempty_users == c.old_union.gt(0).sum()
            assert row.candidate_total_intersection == c.intersection.sum()
            assert row.candidate_total_union == c.union.sum()
    return {'user_comparisons_checked':len(comparators), 'seed_change_comparisons_checked':len(seed_changes),
            'historical_subgroup_means_checked':len(groups), 'top_overlap_seed_comparisons_checked':len(expected_top),
            'candidate_budgets_and_all_users_checked':len(reference)}


def _verify_depth_outputs(frame, run, out, masks, protocol):
    dest = out / 'depth'; old = run / 'confirmation'
    records = []; ledgers = []; rank_records = []; old_checks = 0
    for seed in protocol['seeds']:
        score = pd.read_parquet(old / 'full' / str(seed) / 'scores.parquet')
        np.testing.assert_array_equal(score[KEYS].to_numpy(), frame[KEYS].to_numpy())
        ranks = {name:ranked_rows(frame, score.score.to_numpy(), mask) for name,mask in masks.items()}
        metadata = frame[KEYS + ['orders_since_purchase','purchase_count']].copy()
        metadata['eligibility'] = np.where(masks['dual'],'retained','excluded')
        targets = ranks['full'].loc[ranks['full'].label.eq(1)].merge(metadata,on=KEYS,validate='one_to_one')
        targets['product_group'] = ['recent' if age < 5 else 'dormant_repeat' if count >= 3 else 'dormant_low_frequency' for age,count in zip(targets.orders_since_purchase,targets.purchase_count)]
        targets['rank_band'] = ['1-10' if rank <= 10 else '11-20' if rank <= 20 else '21-50' if rank <= 50 else '>50' for rank in targets['rank']]
        counts = targets.groupby(['eligibility','product_group','rank_band']).size()
        grid = pd.MultiIndex.from_product([['excluded','retained'],['recent','dormant_repeat','dormant_low_frequency'],['1-10','11-20','21-50','>50']],names=['eligibility','product_group','rank_band'])
        rank_records.append(counts.reindex(grid,fill_value=0).rename('targets').reset_index().assign(seed=seed))
        assert counts.sum() == frame.label.sum()
        for k in protocol['depth_k']:
            for name, ranked in ranks.items():
                values = per_at_k(frame, ranked, k)
                records.append(values.reset_index().assign(k=k,policy=name,seed=seed))
                if k == 10:
                    previous = pd.read_parquet(old / name / str(seed) / 'users.parquet').set_index('user_id')
                    _compare_frame(previous, values)
                    old_checks += 1
                elif k in [5,20]:
                    previous = pd.read_csv(old / 'k_sensitivity.csv')
                    row = previous.loc[previous.k.eq(k) & previous.policy.eq(name) & previous.seed.eq(seed)].iloc[0]
                    for field in ['recall','ndcg','precision','fixed_budget_precision','f1','oracle_recall']:
                        _near(row[f'{field}_at_{k}'], values[field].mean())
                    old_checks += 1
            ledger = pruning_ledger(frame, ranks['full'], ranks['dual'], k).rename(columns={'removed_true_hits':'deleted_true_hits','removed_true_recall':'deleted_true_recall'})
            ledgers.append(ledger.reset_index().assign(k=k,seed=seed))
    seed_values = pd.concat(records,ignore_index=True)
    _compare_frame(pd.read_parquet(dest / 'seed_users.parquet').set_index(['k','policy','seed','user_id']), seed_values.set_index(['k','policy','seed','user_id']))
    values = seed_values.drop(columns='seed').groupby(['k','policy','user_id']).mean(numeric_only=True)
    _compare_frame(pd.read_parquet(dest / 'users.parquet').set_index(['k','policy','user_id']), values)
    for filename, expected, keys in [('seed_metrics.csv',seed_values,['k','policy','seed']),('curve_metrics.csv',values.reset_index(),['k','policy'])]:
        saved = pd.read_csv(dest / filename).set_index(keys)
        for group, table in expected.groupby(keys):
            row = saved.loc[group]
            for field in ['recall','ndcg','precision','fixed_budget_precision','coverage','oracle_recall']:
                _near(row[field], table[field].mean())
            _near(row.mean_returned, table.recommended.mean()); _near(row.mean_candidates, table.candidates.mean())
            assert row.users == 8000 and row.nonempty_users == 7472 and row.empty_repeat_targets == 528
    ledger_seed = pd.concat(ledgers,ignore_index=True)
    _compare_frame(pd.read_parquet(dest / 'ledger_seed_users.parquet').set_index(['k','seed','user_id']),ledger_seed.set_index(['k','seed','user_id']))
    ledger = ledger_seed.drop(columns='seed').groupby(['k','user_id']).mean(numeric_only=True)
    _compare_frame(pd.read_parquet(dest / 'ledger_users.parquet').set_index(['k','user_id']),ledger)
    saved_ledger = pd.read_csv(dest / 'rank_ledger.csv').set_index('k')
    for k in protocol['depth_k']:
        table = ledger.loc[k]; row = saved_ledger.loc[k]; nonempty = table.loc[table.repeat_targets.gt(0)]
        for field in ['promoted_true_hits','deleted_true_hits','delta_hits','promoted_true_recall','deleted_true_recall','delta_recall']:
            _near(row[field],nonempty[field].mean())
        change = sign_summary(nonempty.delta_recall)
        for field in ['improved_users','worse_users','unchanged_users','nonempty_users']:
            assert row[field] == change[field]
        for field in ['improved','worse','unchanged']:
            _near(row[f'{field}_fraction'],change[f'{field}_users']/change['nonempty_users'])
        _near(table.delta_recall,values.loc[(k,'dual')].recall-values.loc[(k,'full')].recall)
    rank_seed = pd.concat(rank_records,ignore_index=True)
    rank_keys = ['eligibility','product_group','rank_band']
    _compare_frame(pd.read_csv(dest / 'target_rank_seed_counts.csv').set_index(rank_keys+['seed']),rank_seed.set_index(rank_keys+['seed']))
    rank_mean = rank_seed.drop(columns='seed').groupby(rank_keys).mean()
    saved_ranks = pd.read_csv(dest / 'excluded_target_ranks.csv').set_index(rank_keys)
    _compare_frame(saved_ranks,rank_mean)
    _near(saved_ranks.share_all_repeat_targets,rank_mean.targets/frame.label.sum())
    for eligibility, sub in saved_ranks.groupby(level='eligibility'):
        _near(sub.share_within_eligibility,sub.targets/sub.targets.sum())
    contrasts = pd.read_csv(dest / 'contrasts.csv').set_index(['k','comparator'])
    assert len(contrasts) == 12
    for k in protocol['depth_k']:
        for name in protocol['depth_comparators']:
            differences = values.loc[(k,'dual')].recall-values.loc[(k,name)].recall
            stats = _bootstrap(differences,protocol['depth_bootstrap_replicates'],protocol['bootstrap_seed'],12)
            for field,value in stats.items():_near(contrasts.loc[(k,name),field],value)
    summary = _read(dest / 'summary.json')
    assert summary['posthoc'] is True and summary['model_fits'] == 0 and summary['selection_changed'] is False
    assert summary['old_seed_policy_k_records_reproduced'] == old_checks == 27
    assert summary['repeat_targets'] == frame.label.sum()
    assert summary['excluded_repeat_targets'] == frame.loc[~masks['dual']].label.sum()
    return {'depth_seed_policy_configurations_checked':len(records), 'old_seed_policy_k_records_reproduced':old_checks,
            'depth_ledger_user_seed_k_sets_checked':len(ledgers), 'target_rank_partitions_checked':len(rank_records),
            'depth_contrasts_independently_bootstrapped':len(contrasts)}


def _compare_frame(actual, expected):
    actual, expected = actual.sort_index(), expected.sort_index()
    assert not actual.index.has_duplicates and not expected.index.has_duplicates
    np.testing.assert_array_equal(actual.index.to_numpy(),expected.index.to_numpy())
    for column in expected:
        _near(actual[column],expected[column],column)


def _bootstrap(values, count, seed, family=1):
    values = pd.Series(values).dropna().to_numpy()
    rng = np.random.default_rng(seed); means = np.empty(count)
    for start in range(0,count,100):
        length = min(100,count-start)
        means[start:start+length] = values[rng.integers(len(values),size=(length,len(values)))].mean(axis=1)
    points = np.quantile(means,[.025,.975,.025/family,1-.025/family])
    return dict(users=len(values),delta=float(values.mean()),ci95_low=float(points[0]),ci95_high=float(points[1]),sim_low=float(points[2]),sim_high=float(points[3]))


def _check_change(row, values, n_boot=0, seed=20261116):
    raw = pd.Series(values)
    valid = raw.dropna().copy(); valid.loc[valid.abs().le(1e-12)] = 0.
    stats = sign_summary(valid)
    assert row.users == len(raw) and row.empty_target_users == raw.isna().sum()
    for saved, field in [('nonempty_users','nonempty_users'),('improved_users','improved_users'),('harmed_users','worse_users'),('unchanged_users','unchanged_users'),('mean_delta','mean_delta'),('mean_abs_delta','mean_absolute_delta'),('gain_contribution','gain_contribution')]:
        _near(row[saved],stats[field])
    _near(row.loss_contribution,-stats['loss_contribution'])
    for saved, field in [('improved_fraction','improved_users'),('harmed_fraction','worse_users'),('unchanged_fraction','unchanged_users')]:
        _near(row[saved],stats[field]/len(valid) if len(valid) else np.nan)
    _near(row.changed_mean_delta,valid.loc[valid.ne(0)].mean())
    _near(row.improved_mean_gain,valid.loc[valid.gt(0)].mean())
    _near(row.harmed_mean_loss,-valid.loc[valid.lt(0)].mean())
    if n_boot:
        boot = _bootstrap(valid,n_boot,seed)
        _near(row.ci95_low,boot['ci95_low']); _near(row.ci95_high,boot['ci95_high'])
