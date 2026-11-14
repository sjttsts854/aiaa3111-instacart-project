"""Statistics declared before the third confirmation group is scored."""
import numpy as np
import pandas as pd
from basketlab.metrics import evaluate
from research.analysis import paired_interval
from research.data import save_json
from study3.history import analyze_history
from study3.quota import quota_scores

def analyze(frame,cache,run):
    from study3.run import read,seeds,saved_scores
    conf=read(run/'freeze.json')['protocol'];sel=read(run/'selection.json');dest=run/'confirmation'
    def per(n):return pd.read_parquet(dest/n/'users.parquet')
    contrasts=[]
    for a,b in conf['primary_contrasts']:
        contrasts.append({'left':a,'right':b,**paired_interval(per(a),per(b),conf['bootstrap_replicates'],family=len(conf['primary_contrasts']))})
    pd.DataFrame(contrasts).to_csv(dest/'primary_contrasts.csv',index=False)
    for name in conf['history_models']:
        analyze_history(per(name),frame,dest/f'history_{name}',dataset_label='fresh_v3_'+name)
    diagnostic=[];denom=frame.groupby('user_id').label.sum();covered=frame[frame.co_narrow_covered.gt(0)]
    covdenom=covered.groupby('user_id').label.sum().reindex(denom.index,fill_value=0)
    base={}
    for name in ['full','co_narrow','quota']:
        user_tables=[]
        for seed in conf['model_seeds']:
            top=pd.read_parquet(dest/name/str(seed)/'top10.parquet').merge(frame[['user_id','product_id','co_narrow_covered']],on=['user_id','product_id'],validate='one_to_one')
            top['covered_hits']=top.label*top.co_narrow_covered
            top['uncovered_hits']=top.label*(1-top.co_narrow_covered)
            tab=top.groupby('user_id')[['covered_hits','uncovered_hits','co_narrow_covered']].sum()
            tab['covered_contribution']=tab.covered_hits/denom.replace(0,np.nan)
            tab['uncovered_contribution']=tab.uncovered_hits/denom.replace(0,np.nan)
            tab['covered_recall']=tab.covered_hits/covdenom.replace(0,np.nan)
            user_tables.append(tab.reset_index())
        base[name]=pd.concat(user_tables).groupby('user_id',as_index=False).mean(numeric_only=True)
    for name in ['co_narrow','quota']:
        row={'name':name,'covered_slots_delta':float(base[name].co_narrow_covered.mean()-base['full'].co_narrow_covered.mean())}
        for col in ['covered_contribution','uncovered_contribution','covered_recall']:
            a=base[name][['user_id',col]].rename(columns={col:'recall'})
            b=base['full'][['user_id',col]].rename(columns={col:'recall'})
            stat=paired_interval(a,b,conf['bootstrap_replicates'],family=6)
            row.update({col+'_'+k:v for k,v in stat.items()})
        row['total_delta']=row['covered_contribution_delta']+row['uncovered_contribution_delta']
        np.testing.assert_allclose(row['total_delta'],(per(name).set_index('user_id').recall-per('full').set_index('user_id').recall).mean(),atol=1e-12)
        diagnostic.append(row)
    pd.DataFrame(diagnostic).to_csv(dest/'rule_decomposition.csv',index=False)
    sensitivity=[]
    for name in ['full','co_narrow','quota','tifu','decay']:
        for seed in seeds(name):
            for k in conf['sensitivity_k']:
                if name=='quota':scores=quota_scores(frame,saved_scores(frame,dest/'full'/str(seed)),saved_scores(frame,dest/'co_narrow'/str(seed)),k=k,weight=sel['quota_weight'])
                else:scores=saved_scores(frame,dest/name/str(seed))
                m,_,_=evaluate(frame,scores,k)
                sensitivity.append({'name':name,'seed':seed,'k':k,'recall':m[f'recall_at_{k}'],'ndcg':m[f'ndcg_at_{k}']})
    pd.DataFrame(sensitivity).to_csv(dest/'k_sensitivity.csv',index=False)
    total=pd.read_parquet(cache/'labels_confirm.parquet').groupby('user_id').size()
    full=per('full').set_index('user_id')
    save_json(dest/'scope.json',{'users':len(full),'nonempty_users':int(full.recall.notna().sum()),
        'repeat_target_candidate_coverage':float((full.repeat_targets/total).mean()),
        'whole_basket_recall':float((full.hits/total).mean()),'covered_candidate_fraction':float(frame.co_narrow_covered.mean()),
        'covered_target_fraction':float(covered.label.sum()/frame.label.sum())})
