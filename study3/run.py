"""Bounded extension: development -> frozen fresh-user confirmation."""
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
from basketlab.metrics import attach_labels,evaluate
from research.models import fit,predict
from research.data import save_json,sha
from research.experiment import assert_hashes
from study3.quota import quota_scores

CACHE=Path('data/processed/study3'); RUN=Path('autoresearch/research-v3')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def cfg():return read('study3/protocol.json')
def stamp():return datetime.now(timezone.utc).isoformat()
def seeds(name):return cfg()['model_seeds'] if name in cfg()['learned_models']+['quota'] else [42]

def record(frame,scores,dest,artifact=None):
    dest.mkdir(parents=True,exist_ok=True)
    m,per,top=evaluate(frame,scores,10)
    pd.DataFrame({'user_id':frame.user_id,'product_id':frame.product_id,'score':scores}).to_parquet(dest/'scores.parquet',index=False)
    per.to_parquet(dest/'users.parquet',index=False);top.to_parquet(dest/'top10.parquet',index=False)
    if artifact is not None:joblib.dump(artifact,dest/'model.joblib')
    save_json(dest/'metrics.json',m)
    return m['recall_at_10']

def saved_scores(frame,d):
    s=pd.read_parquet(d/'scores.parquet')
    assert np.array_equal(frame[['user_id','product_id']].to_numpy(),s[['user_id','product_id']].to_numpy())
    return s.score.to_numpy()

def aggregate(part,names):
    rows=[]
    for name in names:
        per=[]
        for seed in seeds(name):
            d=RUN/part/name/str(seed)
            rows.append({'name':name,'seed':seed,**read(d/'metrics.json')})
            per.append(pd.read_parquet(d/'users.parquet'))
        pd.concat(per).groupby('user_id',as_index=False).mean(numeric_only=True).to_parquet(RUN/part/name/'users.parquet',index=False)
    df=pd.DataFrame(rows);df.to_csv(RUN/part/'seed_metrics.csv',index=False)
    summary=df.groupby('name',sort=False).mean(numeric_only=True).drop(columns='seed')
    summary.to_csv(RUN/part/'metrics.csv')
    return summary

def develop():
    if (RUN/'freeze.json').exists():raise RuntimeError('Frozen study; development closed')
    conf=cfg();RUN.mkdir(parents=True,exist_ok=True)
    code_paths=['study3/quota.py','study3/tifu.py','research/models.py','basketlab/metrics.py','research/data.py','research/patterns.py','basketlab/data.py']
    code_identity=RUN/'development_code.json'
    if code_identity.exists():assert_hashes(read(code_identity))
    else:save_json(code_identity,{p:sha(p) for p in code_paths})
    frame=pd.read_parquet(CACHE/'features.parquet').drop(columns='product_name')
    train=attach_labels(frame[frame.split.eq('train')],pd.read_parquet(CACHE/'labels_train.parquet'))
    valid=attach_labels(frame[frame.split.eq('valid')],pd.read_parquet(CACHE/'labels_valid.parquet'))
    del frame
    identity={str(p):sha(p) for p in [Path('study3/protocol.json'),CACHE/'features.parquet',CACHE/'splits.csv',CACHE/'labels_train.parquet',CACHE/'labels_valid.parquet']}
    if (RUN/'development_inputs.json').exists():assert_hashes(read(RUN/'development_inputs.json'))
    else:save_json(RUN/'development_inputs.json',identity)
    for name in ['frequency','decay']+conf['learned_models']:
        for seed in seeds(name):
            d=RUN/'validation'/name/str(seed)
            if (d/'metrics.json').exists():continue
            artifact=fit(train,name,seed);value=record(valid,predict(valid,artifact),d,artifact)
            print(f'VALID {name} {seed}: {value:.6f}',flush=True)
    log=[]
    for weight in conf['quota_weights']:
        values=[]
        for seed in conf['model_seeds']:
            base=saved_scores(valid,RUN/'validation/full'/str(seed));rule=saved_scores(valid,RUN/'validation/co_narrow'/str(seed))
            score=quota_scores(valid,base,rule,weight=weight)
            value,_,_=evaluate(valid,score,10);values.append(value['recall_at_10'])
        log.append({'weight':weight,'recall':float(np.mean(values))})
    best_weight=sorted(log,key=lambda r:(-r['recall'],r['weight']))[0]['weight']
    pd.DataFrame(log).to_csv(RUN/'quota_validation.csv',index=False)
    for seed in conf['model_seeds']:
        score=quota_scores(valid,saved_scores(valid,RUN/'validation/full'/str(seed)),saved_scores(valid,RUN/'validation/co_narrow'/str(seed)),weight=best_weight)
        record(valid,score,RUN/'validation/quota'/str(seed))
    print(f'QUOTA selected weight={best_weight}',flush=True)
    from study3.tifu import fit as fit_tifu,components
    history=pd.read_parquet(CACHE/'history.parquet');splits=pd.read_csv(CACHE/'splits.csv')
    train_ids=splits.loc[splits.split.eq('train'),'user_id']
    query=history[history.user_id.isin(valid.user_id)]
    n_items=int(pd.read_parquet(CACHE/'products.parquet').product_id.max())+1
    tlog=[]
    for i,settings in enumerate(conf['tifu_temporal_grid']):
        artifact=fit_tifu(history,train_ids,settings,n_items=n_items)
        joblib.dump(artifact,RUN/f'tifu_model_{i}.joblib')
        parts=components(query,valid,artifact,neighbor_counts=tuple(conf['tifu_neighbors']))
        np.savez_compressed(RUN/f'tifu_validation_parts_{i}.npz',personal=parts['personal'],**{f'k{k}':v for k,v in parts['neighbors'].items()})
        save_json(RUN/f'tifu_diagnostics_{i}.json',parts['diagnostics'])
        for k in conf['tifu_neighbors']:
            for alpha in conf['tifu_alpha']:
                scores=alpha*parts['personal']+(1-alpha)*parts['neighbors'][k]
                m,_,_=evaluate(valid,scores,10)
                tlog.append({'settings_index':i,'neighbors':k,'alpha':alpha,'recall':m['recall_at_10']})
        print(f'TIFU settings {i}: complete',flush=True)
    choice=sorted(tlog,key=lambda r:(-r['recall'],-r['alpha'],r['neighbors'],r['settings_index']))[0]
    pd.DataFrame(tlog).to_csv(RUN/'tifu_validation.csv',index=False)
    p=np.load(RUN/f"tifu_validation_parts_{choice['settings_index']}.npz")
    record(valid,choice['alpha']*p['personal']+(1-choice['alpha'])*p[f"k{choice['neighbors']}"],RUN/'validation/tifu/42')
    record(valid,p['personal'],RUN/'validation/tifu_self/42')
    selection={'quota_weight':best_weight,'tifu':choice,'tifu_settings':conf['tifu_temporal_grid'][choice['settings_index']],
               'selected_before_confirmation':True,'utc':stamp()}
    save_json(RUN/'selection.json',selection)
    aggregate('validation',['frequency','decay']+conf['learned_models']+['quota','tifu','tifu_self'])
    save_json(RUN/'development_completed.json',{'utc':stamp(),'protocol':conf})
    return selection

def freeze():
    if (RUN/'freeze.json').exists():raise RuntimeError('Already frozen')
    assert read(RUN/'development_completed.json')['protocol']==cfg()
    assert_hashes(read(RUN/'development_inputs.json'))
    assert_hashes(read(RUN/'development_code.json'))
    sources=list(Path('study3').glob('*.py'))+list(Path('study3_tests').glob('*.py'))+list(Path('research').glob('*.py'))+list(Path('basketlab').glob('*.py'))
    sources=[p for p in sources if p.name not in ['report.py','reporting.py','audit.py','auditing.py']]
    paths=sources+[Path('study3/protocol.json'),RUN/'selection.json',RUN/'tifu_validation.csv',RUN/'quota_validation.csv',RUN/'development_code.json',RUN/'development_inputs.json']
    paths+=list(CACHE.glob('*.parquet'))+list(CACHE.glob('*.json'))+[CACHE/'splits.csv',Path(cfg()['excluded_users'])]
    paths+=list(RUN.glob('tifu_model_*.joblib'))+list((RUN/'validation').rglob('model.joblib'))
    save_json(RUN/'freeze.json',{'utc':stamp(),'protocol':cfg(),'selection':read(RUN/'selection.json'),'sha256':{str(p):sha(p) for p in paths}})
    return {'frozen_files':len(paths)}

def confirm():
    if (RUN/'confirmation/completed.json').exists():raise RuntimeError('Confirmation completed')
    frozen=read(RUN/'freeze.json');assert_hashes(frozen['sha256']);conf=frozen['protocol'];selection=frozen['selection']
    frame=pd.read_parquet(CACHE/'features.parquet');frame=frame[frame.split.eq('confirm')]
    frame=attach_labels(frame,pd.read_parquet(CACHE/'labels_confirm.parquet'))
    (RUN/'confirmation').mkdir(exist_ok=True)
    start=RUN/'confirmation/started.json'
    if start.exists():assert read(start)['freeze_sha256']==sha(RUN/'freeze.json')
    else:save_json(start,{'utc':stamp(),'freeze_sha256':sha(RUN/'freeze.json')})
    for name in ['frequency','decay']+conf['learned_models']:
        for seed in seeds(name):
            d=RUN/'confirmation'/name/str(seed)
            if (d/'metrics.json').exists():continue
            artifact=joblib.load(RUN/'validation'/name/str(seed)/'model.joblib')
            value=record(frame,predict(frame,artifact),d)
            print(f'CONFIRM {name} {seed}: {value:.6f}',flush=True)
    for seed in conf['model_seeds']:
        base=saved_scores(frame,RUN/'confirmation/full'/str(seed));rules=saved_scores(frame,RUN/'confirmation/co_narrow'/str(seed))
        record(frame,quota_scores(frame,base,rules,weight=selection['quota_weight']),RUN/'confirmation/quota'/str(seed))
    from study3.tifu import components
    history=pd.read_parquet(CACHE/'history.parquet');query=history[history.user_id.isin(frame.user_id)]
    chosen=selection['tifu'];artifact=joblib.load(RUN/f"tifu_model_{chosen['settings_index']}.joblib")
    parts=components(query,frame,artifact,neighbor_counts=(chosen['neighbors'],))
    record(frame,chosen['alpha']*parts['personal']+(1-chosen['alpha'])*parts['neighbors'][chosen['neighbors']],RUN/'confirmation/tifu/42')
    record(frame,parts['personal'],RUN/'confirmation/tifu_self/42')
    aggregate('confirmation',['frequency','decay']+conf['learned_models']+['quota','tifu','tifu_self'])
    from study3.analysis import analyze
    analyze(frame,CACHE,RUN)
    save_json(RUN/'confirmation/completed.json',{'utc':stamp(),'freeze_sha256':sha(RUN/'freeze.json')})
    return read(RUN/'selection.json')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['develop','freeze','confirm','audit','report'])
    args=parser.parse_args()
    if args.stage=='audit':
        from study3.audit import verify
        result=verify(CACHE,RUN)
    elif args.stage=='report':
        from study3.report import report
        result=report(CACHE,RUN)
    else:result=globals()[args.stage]()
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str))

if __name__=='__main__':main()
