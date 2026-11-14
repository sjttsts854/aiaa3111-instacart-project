"""Frozen study of ranking-information windows and eligible candidate pools."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from basketlab.metrics import attach_labels
from research.models import fit,predict
from research.data import sha,save_json
from research.experiment import assert_hashes
from research.analysis import paired_interval
from study4.memory import add_window_scores,candidate_mask,evaluate_fixed

CACHE=Path('data/processed/study4');RUN=Path('autoresearch/research-v4')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def cfg():return read('study4/protocol.json')
def stamp():return datetime.now(timezone.utc).isoformat()
def sources():
    return [p for p in list(Path('study4').glob('*.py'))+list(Path('study4_tests').glob('*.py')) if p.name not in ['report.py','audit.py']]+[Path(p) for p in ['study4/protocol.json','research/models.py','research/data.py','research/analysis.py','basketlab/data.py','basketlab/metrics.py','study3/history.py']]

def record(frame,scores,mask,path,k=10,artifact=None,save_scores=False):
    path.mkdir(parents=True,exist_ok=True)
    metrics,per,top=evaluate_fixed(frame,scores,mask,k=k)
    per.to_parquet(path/'users.parquet',index=False);top.to_parquet(path/'top10.parquet',index=False)
    if save_scores:
        pd.DataFrame({'user_id':frame.user_id,'product_id':frame.product_id,'score':scores}).to_parquet(path/'scores.parquet',index=False)
    if artifact is not None:joblib.dump(artifact,path/'model.joblib')
    save_json(path/'metrics.json',metrics)
    return metrics,per,top

def scores_at(frame,path):
    values=pd.read_parquet(path/'scores.parquet')
    if not np.array_equal(frame[['user_id','product_id']].to_numpy(),values[['user_id','product_id']].to_numpy()):raise ValueError('Score rows changed')
    return values.score.to_numpy()

def aggregate(part,names,seeds):
    rows=[]
    for name in names:
        tables=[]
        for seed in seeds:
            path=RUN/part/name/str(seed)
            tables.append(pd.read_parquet(path/'users.parquet'))
            rows.append({'name':name,'seed':seed,**read(path/'metrics.json')})
        pd.concat(tables).groupby('user_id',as_index=False).mean(numeric_only=True).to_parquet(RUN/part/name/'users.parquet',index=False)
    df=pd.DataFrame(rows);df.to_csv(RUN/part/'seed_metrics.csv',index=False)
    result=df.groupby('name',sort=False).mean(numeric_only=True).drop(columns='seed')
    result.to_csv(RUN/part/'metrics.csv');return result

def masks(frame,selection):
    full=candidate_mask(frame,'all');recent=candidate_mask(frame,'recent',window=5)
    chosen=candidate_mask(frame,**selection)
    budgets=pd.Series(chosen,index=frame.index).groupby(frame.user_id).sum().astype(int)
    return {'full':full,'recent':recent,'dual':chosen,
            'frequency_budget':candidate_mask(frame,'pure_frequency_budget',budget=budgets),
            'recency_budget':candidate_mask(frame,'pure_recency_budget',budget=budgets)}

def develop():
    conf=cfg();RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'freeze.json').exists():raise RuntimeError('Development closed')
    identity=RUN/'development_inputs.json'
    if identity.exists():assert_hashes(read(identity))
    else:
        paths=sources()+[CACHE/'candidates.parquet',CACHE/'labels_train.parquet',CACHE/'labels_valid.parquet',CACHE/'splits.csv']
        save_json(identity,{str(p):sha(p) for p in paths})
    data=pd.read_parquet(CACHE/'candidates.parquet').drop(columns='product_name')
    train=attach_labels(data[data.split.eq('train')],pd.read_parquet(CACHE/'labels_train.parquet'))
    valid=attach_labels(data[data.split.eq('valid')],pd.read_parquet(CACHE/'labels_valid.parquet'))
    all_mask=np.ones(len(valid),dtype=bool)
    for seed in conf['model_seeds']:
        path=RUN/'validation/full'/str(seed)
        if not (path/'metrics.json').exists():
            model=fit(train,'full',seed)
            m,_,_=record(valid,predict(valid,model),all_mask,path,artifact=model,save_scores=True)
            print(f"VALID full {seed}: {m['recall_at_10']:.6f}",flush=True)
    aggregate('validation',['full'],conf['model_seeds'])
    baseline=pd.read_parquet(RUN/'validation/full/users.parquet');grid=[]
    for old_count in conf['dual_old_counts']:
        mask=candidate_mask(valid,'dual',window=5,m=old_count);tables=[]
        for seed in conf['model_seeds']:
            _,per,_=evaluate_fixed(valid,scores_at(valid,RUN/'validation/full'/str(seed)),mask,10)
            tables.append(per)
        per=pd.concat(tables).groupby('user_id',as_index=False).mean(numeric_only=True)
        stats=paired_interval(per,baseline,conf['bootstrap_replicates'],family=1)
        count=float(pd.Series(mask).groupby(valid.user_id.to_numpy()).sum().mean())
        grid.append({'old_count':old_count,'mean_candidates':count,'recall':float(per.recall.mean()),**stats,
                     'eligible':bool(stats['ci95_low']>=-conf['practical_margin'])})
        print(f"VALID dual M={old_count}: {per.recall.mean():.6f}, candidates={count:.2f}, eligible={grid[-1]['eligible']}",flush=True)
    pd.DataFrame(grid).to_csv(RUN/'policy_validation.csv',index=False)
    passing=[r for r in grid if r['eligible']]
    if passing:
        best=sorted(passing,key=lambda r:(r['mean_candidates'],-r['recall'],r['old_count']))[0]
        policy={'kind':'dual','window':5,'m':best['old_count']}
    else:policy={'kind':'all'}
    save_json(RUN/'selection.json',{'policy':policy,'utc':stamp(),'criterion':conf['policy_selection'],'selected_before_confirmation':True})
    for name,mask in masks(valid,policy).items():
        if name=='full':continue
        for seed in conf['model_seeds']:record(valid,scores_at(valid,RUN/'validation/full'/str(seed)),mask,RUN/'validation'/name/str(seed))
    aggregate('validation',list(masks(valid,policy)),conf['model_seeds'])
    save_json(RUN/'development_completed.json',{'utc':stamp(),'protocol':conf})
    return read(RUN/'selection.json')

def freeze():
    if (RUN/'freeze.json').exists():raise RuntimeError('Already frozen')
    assert_hashes(read(RUN/'development_inputs.json'))
    if read(RUN/'development_completed.json')['protocol']!=cfg():raise ValueError('Protocol changed')
    paths=sources()+list(CACHE.glob('*.parquet'))+list(CACHE.glob('*.json'))+[CACHE/'splits.csv',Path(cfg()['excluded_users']),RUN/'selection.json',RUN/'policy_validation.csv',RUN/'development_inputs.json',RUN/'development_completed.json']+[p for p in (RUN/'validation').rglob('*') if p.is_file()]
    save_json(RUN/'freeze.json',{'utc':stamp(),'protocol':cfg(),'selection':read(RUN/'selection.json'),'sha256':{str(p):sha(p) for p in paths}})
    return {'frozen_files':len(paths)}

def confirm():
    if (RUN/'confirmation/completed.json').exists():raise RuntimeError('Already confirmed')
    frozen=read(RUN/'freeze.json');assert_hashes(frozen['sha256']);conf=frozen['protocol']
    frame=pd.read_parquet(CACHE/'candidates.parquet');frame=frame[frame.split.eq('confirm')]
    history=pd.read_parquet(CACHE/'history.parquet');history=history[history.user_id.isin(frame.user_id)]
    frame=add_window_scores(frame,history,windows=(3,5,10),decay=.8)
    frame=attach_labels(frame,pd.read_parquet(CACHE/'labels_confirm.parquet'))
    dest=RUN/'confirmation';dest.mkdir(exist_ok=True)
    if (dest/'started.json').exists():assert read(dest/'started.json')['freeze_sha256']==sha(RUN/'freeze.json')
    else:save_json(dest/'started.json',{'utc':stamp(),'freeze_sha256':sha(RUN/'freeze.json')})
    choices=masks(frame,frozen['selection']['policy'])
    for seed in conf['model_seeds']:
        model=joblib.load(RUN/'validation/full'/str(seed)/'model.joblib');scores=predict(frame,model)
        for name,mask in choices.items():
            m,_,_=record(frame,scores,mask,dest/name/str(seed),save_scores=name=='full')
            print(f"CONFIRM {name} {seed}: {m['recall_at_10']:.6f}",flush=True)
    aggregate('confirmation',list(choices),conf['model_seeds'])
    from study4.analysis import analyze
    analyze(frame,CACHE,RUN)
    save_json(dest/'completed.json',{'utc':stamp(),'freeze_sha256':sha(RUN/'freeze.json')})
    return {'status':'confirmation complete','selection':frozen['selection']}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['develop','freeze','confirm','audit','report'])
    arg=parser.parse_args().stage
    if arg=='audit':
        from study4.audit import verify
        result=verify(CACHE,RUN)
    elif arg=='report':
        from study4.report import report
        result=report(CACHE,RUN)
    else:result=globals()[arg]()
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__':main()
