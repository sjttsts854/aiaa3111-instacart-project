"""Predeclared analysis of fixed targets, score windows and candidate expansion."""
from pathlib import Path
import numpy as np
import pandas as pd
from research.analysis import paired_interval
from research.data import save_json
from study4.memory import candidate_mask,evaluate_fixed,expansion_ledger

def linear_contrast(tables, coefficients):
    """Combine paired per-user recalls without silently dropping zero-hit users."""
    reference=tables[0].set_index('user_id').recall.sort_index()
    value=reference*coefficients[0]
    for table,coefficient in zip(tables[1:],coefficients[1:]):
        series=table.set_index('user_id').recall.sort_index()
        if not series.index.equals(reference.index):raise ValueError('Contrast user universes differ')
        if not series.isna().equals(reference.isna()):raise ValueError('Contrast target denominators differ')
        value=value+coefficient*series
    return value.rename('recall').reset_index()

def analyze(frame,cache,run):
    from study4.run import read,record,scores_at,masks
    from study4.history import analyze_observed
    from study3.history import analyze_history
    config=read(run/'freeze.json')['protocol'];dest=run/'confirmation'
    seeds=config['model_seeds'];family=len(config['primary_contrasts']);n_boot=config['bootstrap_replicates']
    selection=read(run/'selection.json')['policy']
    manifest={'cells':[],'ledgers':[]};cell_metrics=[];tables={};tops={}
    long_ids=frame.loc[frame.history_orders.ge(16),'user_id'].unique()
    def per(name):return pd.read_parquet(dest/name/'users.parquet')
    def interval(value,subgroup):
        if subgroup=='long_history':value=value[value.user_id.isin(long_ids)]
        zero=value.copy();zero['recall']=0.0
        return paired_interval(value,zero,n_boot=n_boot,family=family)
    for window in [config['core_window']]+config['sensitivity_windows']:
        for method in config['factorial_scoring']:
            for score_window in [str(window),'all']:
                column=(f'{method}_w{window}' if score_window!='all' else {'frequency':'purchase_count','decay':'decay_score'}[method])
                for pool in [str(window),'all']:
                    name=f'{method}_s{score_window}_c{pool}'
                    # all/all is the same fixed cell for every window.
                    if name in tables:continue
                    policy={'kind':'all'} if pool=='all' else {'kind':'recent','window':window}
                    m,p,t=record(frame,frame[column].to_numpy(),candidate_mask(frame,**policy),dest/'factorial'/name)
                    tables[name]=p;tops[name]=t
                    manifest['cells'].append({'path':f'factorial/{name}','score':column,'mask':policy,'k':10,'seed':None})
                    cell_metrics.append({'name':name,'method':method,'score_window':score_window,'pool':pool,'subgroup':'all_nonempty',**m})
                    long=p[p.user_id.isin(long_ids)]
                    cell_metrics.append({'name':name,'method':method,'score_window':score_window,'pool':pool,'subgroup':'long_history',
                                         'recall_at_10':float(long.recall.mean()),'users':len(long),'empty_repeat_targets':int(long.recall.isna().sum()),
                                         'candidate_coverage':float(long.coverage.mean()),'oracle_recall_at_10':float(long.oracle_recall.mean()),
                                         'mean_candidates':float(long.candidates.mean())})
    pd.DataFrame(cell_metrics).to_csv(dest/'factorial_metrics.csv',index=False)
    contrasts=[];sensitivity=[]
    for window in [config['core_window']]+config['sensitivity_windows']:
        for method in config['factorial_scoring']:
            aa=tables[f'{method}_sall_call'];ra=tables[f'{method}_s{window}_call']
            ar=tables[f'{method}_sall_c{window}'];rr=tables[f'{method}_s{window}_c{window}']
            specs={'information':([aa,ra],[1,-1]),'expansion':([aa,ar],[1,-1]),
                   'interaction':([aa,ra,ar,rr],[1,-1,-1,1])}
            for effect,(ts,cs) in specs.items():
                value=linear_contrast(ts,cs)
                stats={'contrast':f'{method}_{effect}','window':window,'subgroup':'long_history',**interval(value,'long_history')}
                (contrasts if window==config['core_window'] else sensitivity).append(stats)
                sensitivity.append({'contrast':f'{method}_{effect}','window':window,'subgroup':'all_nonempty',**interval(value,'all_nonempty')})
    for control in ['full','frequency_budget','recency_budget']:
        stats=interval(linear_contrast([per('dual'),per(control)],[1,-1]),'all_nonempty')
        contrasts.append({'contrast':f'dual_vs_{control}','window':5,'subgroup':'all_nonempty',**stats})
    primary=pd.DataFrame(contrasts)
    primary['positive_confirmed']=primary.sim_low.gt(0)
    primary['noninferiority_margin']=config['practical_margin']
    primary['noninferiority_confirmed']=primary.sim_low.ge(-config['practical_margin'])
    primary.to_csv(dest/'primary_contrasts.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(dest/'factorial_sensitivity.csv',index=False)
    ledger_summaries=[]
    def ledger(name,small_path,large_path,small_top,large_top,small_mask,seed=None):
        values=expansion_ledger(frame,small_top,large_top,small_mask,10)
        path=dest/'ledgers'/name;path.parent.mkdir(parents=True,exist_ok=True);values.to_parquet(path.with_suffix('.parquet'),index=False)
        manifest['ledgers'].append({'path':str(path.with_suffix('.parquet').relative_to(dest)).replace('\\','/'),
                                    'small_path':small_path,'large_path':large_path,'small_mask':{'kind':'recent','window':5},'k':10})
        for group in ['all_nonempty','long_history']:
            sub=values if group=='all_nonempty' else values[values.user_id.isin(long_ids)]
            sub=sub[sub.repeat_targets.gt(0)]
            ledger_summaries.append({'name':name,'seed':seed,'subgroup':group,'users':len(sub),
                                     **{c:float(sub[c].mean()) for c in ['new_true_hits','lost_old_true_hits','delta_hits','new_true_recall','lost_old_true_recall','delta_recall']}})
    recent=candidate_mask(frame,'recent',window=5)
    for method in config['factorial_scoring']:
        small=f'{method}_sall_c5';large=f'{method}_sall_call'
        ledger(method,f'factorial/{small}',f'factorial/{large}',tops[small],tops[large],recent)
    for seed in seeds:
        small=pd.read_parquet(dest/'recent'/str(seed)/'top10.parquet')
        for larger in ['full','dual']:
            ledger(f'learned_{larger}_{seed}',f'recent/{seed}',f'{larger}/{seed}',small,
                   pd.read_parquet(dest/larger/str(seed)/'top10.parquet'),recent,seed)
    pd.DataFrame(ledger_summaries).to_csv(dest/'ledger_summary.csv',index=False)
    save_json(dest/'analysis_manifest.json',manifest)
    print('CONFIRM fixed-user factorial and exact expansion ledgers complete',flush=True)
    curves=[]
    scores={seed:scores_at(frame,dest/'full'/str(seed)) for seed in seeds}
    for old_count in config['dual_old_counts']:
        chosen=candidate_mask(frame,'dual',window=5,m=old_count)
        budgets=pd.Series(chosen,index=frame.index).groupby(frame.user_id).sum().astype(int)
        alternatives={'dual':chosen,'frequency_budget':candidate_mask(frame,'pure_frequency_budget',budget=budgets),
                      'recency_budget':candidate_mask(frame,'pure_recency_budget',budget=budgets)}
        for name,mask in alternatives.items():
            for seed in seeds:
                m,_,_=evaluate_fixed(frame,scores[seed],mask,10)
                curves.append({'old_count':old_count,'policy':name,'seed':seed,**m})
    pd.DataFrame(curves).to_csv(dest/'candidate_curves.csv',index=False)
    k_rows=[]
    for k in config['sensitivity_k']:
        for name,mask in masks(frame,selection).items():
            for seed in seeds:
                m,_,_=evaluate_fixed(frame,scores[seed],mask,k)
                k_rows.append({'k':k,'policy':name,'seed':seed,**m})
    pd.DataFrame(k_rows).to_csv(dest/'k_sensitivity.csv',index=False)
    groups=frame[['user_id','product_id','label','orders_since_purchase','purchase_count']].copy()
    groups['product_group']=np.select([groups.orders_since_purchase.lt(5),groups.purchase_count.ge(3)],
                                      ['recent','dormant_repeat'],default='dormant_low_frequency')
    totals=groups.groupby('product_group').agg(candidates=('label','size'),targets=('label','sum'))
    group_rows=[]
    for name in ['full','recent','dual','frequency_budget','recency_budget']:
        mask=masks(frame,selection)[name]
        covered=groups.loc[mask].groupby('product_group').label.sum()
        for seed in seeds:
            t=pd.read_parquet(dest/name/str(seed)/'top10.parquet').merge(groups[['user_id','product_id','product_group']],on=['user_id','product_id'],validate='one_to_one')
            hits=t.groupby('product_group').label.sum();exposure=t.groupby('product_group').size()
            for g,row in totals.iterrows():
                group_rows.append({'policy':name,'seed':seed,'product_group':g,'candidates':int(row.candidates),'targets':int(row.targets),
                                   'covered_targets':int(covered.get(g,0)),'hits':int(hits.get(g,0)),'recommendations':int(exposure.get(g,0)),
                                   'target_share':float(row.targets/totals.targets.sum()),'micro_recall':float(hits.get(g,0)/row.targets) if row.targets else None})
    pd.DataFrame(group_rows).to_csv(dest/'product_group_diagnostics.csv',index=False)
    labels=pd.read_parquet(cache/'labels_confirm.parquet')
    total_targets=labels.groupby('user_id').size().rename('whole_targets')
    scopes=[]
    for name in ['full','recent','dual','frequency_budget','recency_budget']:
        p=per(name).join(total_targets,on='user_id')
        scopes.append({'policy':name,'users':len(p),'repeat_nonempty_users':int(p.repeat_targets.gt(0).sum()),
                       'macro_repeat_recall':float(p.recall.mean()),'macro_whole_basket_recall':float((p.hits/p.whole_targets).mean()),
                       'repeat_target_share_micro':float(p.repeat_targets.sum()/p.whole_targets.sum()),
                       'mean_returned':float(p.recommended.mean())})
    pd.DataFrame(scopes).to_csv(dest/'task_scope.csv',index=False)
    print('CONFIRM budget, K and historical-product diagnostics complete',flush=True)
    observed=analyze_observed(per('full'),frame,dest/'history_observed',n_boot=1000)
    future=analyze_history(per('full'),frame,dest/'history_future_sensitivity',n_boot=1000,seed=20261115,dataset_label='study4_posthoc_future_target_conditioning')
    save_json(dest/'analysis_summary.json',{'primary_contrasts':len(primary),'fixed_user_factorial_cells':len(manifest['cells']),
              'ledger_files':len(manifest['ledgers']),'history_observed':observed,'future_target_conditioning_sensitivity':future,
              'limits':['All ranking-window cells know complete historical catalog membership.','Future target size is excluded from observed-only strata.',
                        'Curves and K sensitivity never change frozen policy.','Intervals condition on models and sample support; no human-behavior causal effects.']})
    print('CONFIRM observed-only history analysis and separate future-target sensitivity complete',flush=True)
