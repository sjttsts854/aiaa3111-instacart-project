"""Build a lean, English source/evidence handoff without modifying old studies."""
from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from study4.memory import candidate_mask

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'final/package'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)

def build():
    DEST.mkdir(parents=True,exist_ok=True)
    for source,target in [('README_PACKAGE.md','README.md'),('evidence_app.py','evidence_app.py'),
                          ('verify_package.py','verify_package.py'),('check_package_demo.py','check_package_demo.py')]:
        copy(ROOT/'closeout'/source,DEST/target)
    (DEST/'requirements-demo.txt').write_text('pandas==2.2.3\npyarrow==19.0.0\nstreamlit==1.45.1\n',encoding='utf-8')
    cache=ROOT/'data/processed/study4';run=ROOT/'autoresearch/research-v4';demo=DEST/'data_demo';demo.mkdir(exist_ok=True)
    frame=pd.read_parquet(cache/'candidates.parquet');frame=frame[frame.split.eq('confirm')].copy()
    selection=json.loads((run/'selection.json').read_text())['policy']
    dual=candidate_mask(frame,**selection)
    budgets=pd.Series(dual,index=frame.index).groupby(frame.user_id).sum().astype(int)
    recency=candidate_mask(frame,'pure_recency_budget',budget=budgets)
    catalog=frame[['user_id','product_id','purchase_count','orders_since_purchase']].copy()
    catalog['dual_eligible']=dual;catalog['recency_eligible']=recency
    catalog.to_parquet(demo/'historical_catalog.parquet',index=False)
    user_columns=['user_id','history_orders','basket_mean','distinct_products']
    frame.groupby('user_id',as_index=False)[user_columns[1:]].first().to_parquet(demo/'users.parquet',index=False)
    targets=pd.read_parquet(cache/'labels_confirm.parquet')
    histkeys=pd.MultiIndex.from_frame(catalog[['user_id','product_id']])
    targets['is_repeat']=pd.MultiIndex.from_frame(targets[['user_id','product_id']]).isin(histkeys)
    targets.to_parquet(demo/'targets.parquet',index=False)
    records=[];metrics={}
    for name in ['full','dual','recency_budget']:
        folder=run/'confirmation'/name/'42'
        records.append(pd.read_parquet(folder/'top10.parquet').assign(policy=name))
        metrics[name]=json.loads((folder/'metrics.json').read_text())
    predictions=pd.concat(records,ignore_index=True)
    predictions.to_parquet(demo/'predictions.parquet',index=False)
    products=pd.read_parquet(cache/'products.parquet')
    needed=set(catalog.product_id)|set(targets.product_id)
    products[products.product_id.isin(needed)][['product_id','product_name']].to_parquet(demo/'products.parquet',index=False)
    dump(demo/'seed42_metrics.json',metrics)
    repeat=targets[targets.is_repeat].groupby('user_id').size().reindex(sorted(frame.user_id.unique()),fill_value=0)
    lists=predictions.sort_values('rank').groupby(['user_id','policy']).product_id.apply(tuple).unstack('policy')
    hits=predictions.groupby(['user_id','policy']).label.sum().unstack('policy')
    dump(demo/'metadata.json',{'source':'Frozen Study 4 confirmation outputs','case_model_seed':42,'users':len(repeat),
          'nonempty_repeat_users':int(repeat.gt(0).sum()),'empty_repeat_users':int(repeat.eq(0).sum()),
          'suggested_ordinary_user':int(repeat[repeat.gt(0)].index[0]),'suggested_empty_user':int(repeat[repeat.eq(0)].index[0]),
          'suggested_changed_list_user':int(lists[lists.full.ne(lists.dual)].index[0]),
          'suggested_improved_user':int(hits[hits.dual.gt(hits.full)].index[0]),
          'suggested_harmed_user':int(hits[hits.dual.lt(hits.full)].index[0]),
          'example_selection':'Smallest anonymous user ID satisfying each condition; illustrative, not a representative sample',
          'selection':selection,'no_model_fitting_in_viewer':True,'user_ids':'Anonymous IDs from the public dataset'})
    for sub in ['study4','study4_followup']:
        for source in (ROOT/'reports'/sub).rglob('*'):
            if source.is_file() and source.suffix.lower() in ['.csv','.json','.png']:
                copy(source,DEST/'evidence'/sub/source.relative_to(ROOT/'reports'/sub))
    for sub,tablepath in [('study2','reports/tables'),('study3','reports/study3/tables')]:
        for source in (ROOT/tablepath).glob('*.csv'):copy(source,DEST/'evidence'/sub/'tables'/source.name)
    folders=['basketlab','research','study3','study4','study4_followup','configs','tests','study3_tests','study4_tests','followup_tests']
    code=DEST/'code'
    for folder in folders:
        for source in (ROOT/folder).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix in ['.py','.json','.md','.txt']:
                copy(source,code/folder/source.relative_to(ROOT/folder))
    for name in ['run.py','requirements.txt','requirements-study-lock.txt','pytest.ini']:
        copy(ROOT/name,code/name)
    for name in ['autoresearch_loop.py','verify_artifacts.py']:copy(ROOT/'scripts'/name,code/'scripts'/name)
    (code/'README.md').write_text('# Research source\n\nRun commands from this directory. See ../REPRODUCE.md for the documented command sequence and ../DATA.md for inputs. Source modules and historical implementation notes are preserved unchanged, including original comments. The new presentation materials and portable viewer are English.\n',encoding='utf-8')
    for label,folder in [('study2','autoresearch/research-v2'),('study3','autoresearch/research-v3'),('study4','autoresearch/research-v4'),('posthoc','autoresearch/research-v4-followup')]:
        for name in ['audit.json','freeze.json','inputs.json','selection.json']:
            source=ROOT/folder/name
            if source.exists():copy(source,DEST/'provenance'/f'{label}_{name}')
    raw_manifest=json.loads((cache/'manifest.json').read_text())['source']
    rows=['| File | Bytes | SHA256 |','| --- | ---: | --- |']+[f"| {name} | {meta['bytes']} | {meta['sha256']} |" for name,meta in raw_manifest.items()]
    (DEST/'DATA.md').write_text('''# Dataset and source identity

Download the Instacart mirror from https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset and extract the following six files into `code/archive/`. You may need a Kaggle account. Review the source terms before redistribution. This package does not redistribute the six raw source files.

The mirror contains the original competition data, including labeled terminal baskets and unlabeled competition test baskets. Research targets use labeled users only. CSV identity is established by the hashes below; the files alone do not establish the mirror webpage revision number.

'''+ '\n'.join(rows)+'''

Source totals: 206,209 users, 3,421,083 orders and 49,688 products; 131,209 users have labeled terminal orders. There are no prices or absolute timestamps; intervals are capped at 30 days. The task predicts next-order repeats within each user's complete observed historical catalog, not forgotten needs or all first-purchase products.

`data_demo/` contains compact derived records for the 8,000 Study 4 confirmation users: product history counts/recency, terminal products with historical membership, frozen seed-42 recommendations, and product names. These public-dataset user IDs are anonymous. This subset is sufficient for inspecting frozen examples, not for retraining.
''',encoding='utf-8')
    (DEST/'REPRODUCE.md').write_text(REPRODUCE,encoding='utf-8')
    report=ROOT/'final/Report_Draft.pdf'
    if report.exists():copy(report,DEST/'Report.pdf')
    finalize_manifest()
    return {'package':str(DEST),'files':sum(p.is_file() for p in DEST.rglob('*')),'bytes':sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),'case_metadata':json.loads((demo/'metadata.json').read_text())}

def finalize_manifest():
    files={str(p.relative_to(DEST)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='package_manifest.json' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    dump(DEST/'package_manifest.json',{'generated_utc':datetime.now(timezone.utc).isoformat(),'status':'team review draft','files':files,
        'integrity_scope':'Hashes identify this portable handoff. Original scientific audits are separate records under provenance/.',
        'excluded':['original raw CSVs','virtual environments','.git','trained model binaries','full processed caches','private credentials']})

REPRODUCE='''# Reproducing the experiments

## Portable inspection

Start with the top-level `verify_package.py` and `evidence_app.py`. They do not need raw data or model fitting. The verifier independently recomputes seed-42 case Recall@10 using full repeat targets, checks candidate budgets, empty targets and all file hashes. Its PASS is a package-level check, not a rerun of the original raw-source audit.

## Unit tests

From `code/`, create an environment and install `requirements-study-lock.txt`. Python 3.13.5 was used for the recorded run. Other supported Python/platform combinations may differ numerically; do not promise byte-identical model artifacts or confidence endpoints.

```powershell
python -m pip install -r requirements-study-lock.txt
python -m pytest tests study3_tests study4_tests followup_tests -q
```

The original workspace passed 108 research tests. These are implementation checks, not independent evidence of scientific generalization.

## Recreate the main Study 4 and its posthoc diagnostics

Run the following commands in a **fresh copy of `code/`**, after placing the six verified source CSVs in `archive/`. The initial three prepare stages reconstruct the previously used user cohorts that Study 4 excludes. They do not require refitting the earlier methods, although `study3.prepare` also builds its fixed rule features. Preparation and fitting read millions of rows, so allow substantial memory and disk space. No runtime budget is promised.

```powershell
python run.py prepare
python -m research.run prepare
python -m study3.prepare
python -m study4.prepare
python -m study4.run develop
python -m study4.run freeze
python -m study4.run confirm
python -m study4.run audit
python -m study4.run report
python -m study4_followup.run analyze
python -m study4_followup.run audit
python -m study4_followup.run report
```

The fixed seeds reproduce the original cohort design. Rerunning these already-seen users is a reproduction, not a new independent confirmation. Development selects M only using validation; confirmation does not reselect it. Followup analysis remains posthoc even when rerun from this source.

Files and models are hashed by each newly created run before its own confirmation. Original `provenance/*freeze.json` records refer to the original complete workspace, whose caches and binaries are not bundled. Do not copy those manifests into a new run or expect their hashes to match refitted model serialization. Do not edit frozen sources after a new freeze. Completed-stage protection is intentional; start another clean reproduction folder instead of deleting guards to retune known confirmation users.

## Optional earlier evidence

Study 2 requires its prepared data above, then `python -m research.run features`, `develop`, `freeze`, `confirm`, `audit`, `report`. Its freeze records a Git commit, so initialize a local Git repository and commit the source before that freeze if the reproduction folder is not already a repository. Use the team's own Git identity. Study 3 then uses `python -m study3.run develop`, `freeze`, `confirm`, `audit`, `report`. The TIFU adaptation and its author-code semantics are documented under `study3/`; the adaptation is not a reproduction of the original paper's whole-basket benchmark scores.

The initial 10,000-user exploratory loop is retained as source (`run.py`, `configs/`, `scripts/autoresearch_loop.py`). It is background to the final question, not required to refit for the Study 4 result. The lean package's top-level English viewer is the supported presentation entry point; the original legacy app was not included because it depends on the large older model caches.

## What was validated for this handoff

- Original studies and posthoc diagnostics have saved independent audit PASS records.
- Original frozen experiment files were not altered for this handoff.
- The packaged viewer uses genuine Study 4 seed-42 outputs, not the legacy Study 2 cases.
- Package metric and integrity checks, portable UI cases and the copied research test suite are checked separately during closeout.
- No clean-room refitting of millions of rows is claimed merely because documentation and source are included.
'''

if __name__=='__main__':print(json.dumps(build(),indent=2))
