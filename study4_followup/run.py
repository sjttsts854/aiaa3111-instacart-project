"""Bounded posthoc diagnostics: read frozen predictions, never fit or select."""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
from research.data import sha,save_json
from research.experiment import assert_hashes

CACHE=Path('data/processed/study4')
RUN=Path('autoresearch/research-v4')
OUT=Path('autoresearch/research-v4-followup')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def stamp():return datetime.now(timezone.utc).isoformat()

def inputs():
    files=[CACHE/'candidates.parquet',CACHE/'labels_confirm.parquet',RUN/'freeze.json',RUN/'selection.json',
           RUN/'audit.json',RUN/'confirmation/completed.json',RUN/'confirmation/metrics.csv',
           RUN/'confirmation/k_sensitivity.csv',RUN/'confirmation/primary_contrasts.csv']
    for name in ['full','dual','recent','frequency_budget','recency_budget']:
        files.append(RUN/'confirmation'/name/'users.parquet')
        for seed in [42,73,123]:
            folder=RUN/'confirmation'/name/str(seed)
            files.extend(folder/p for p in ['users.parquet','top10.parquet','metrics.json'])
            if name=='full':files.append(folder/'scores.parquet')
    files+=list(Path('study4_followup').glob('*.py'))+list(Path('followup_tests').glob('*.py'))+[Path('study4_followup/protocol.json')]
    return sorted(set(p for p in files if p.name not in ['report.py','audit.py']),key=str)

def analyze():
    if (OUT/'completed.json').exists():raise RuntimeError('Posthoc diagnostic run already complete; use audit/report only')
    if read(RUN/'audit.json')['status']!='PASS':raise RuntimeError('Original study audit must pass')
    assert_hashes(read(RUN/'freeze.json')['sha256'])
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'inputs.json'
    if manifest.exists():assert_hashes(read(manifest)['sha256'])
    else:save_json(manifest,{'utc':stamp(),'posthoc':True,'source_confirmation_completed':read(RUN/'confirmation/completed.json'),
                             'protocol':read('study4_followup/protocol.json'),'sha256':{str(p):sha(p) for p in inputs()}})
    from study4_followup.users import analyze_users
    from study4_followup.depth import analyze_depth
    users=analyze_users(CACHE,RUN,OUT)
    print('POSTHOC user changes and same-budget overlap complete',flush=True)
    depth=analyze_depth(CACHE,RUN,OUT)
    print('POSTHOC list-depth and target-rank decomposition complete',flush=True)
    assert_hashes(read(manifest)['sha256'])
    save_json(OUT/'completed.json',{'utc':stamp(),'posthoc':True,'input_manifest_sha256':sha(manifest),'users':users,'depth':depth})
    return {'status':'complete','posthoc':True,'output':str(OUT)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['analyze','audit','report'])
    stage=parser.parse_args().stage
    if stage=='analyze':result=analyze()
    elif stage=='audit':
        from study4_followup.audit import verify
        result=verify(CACHE,RUN,OUT)
    else:
        from study4_followup.report import report
        result=report(CACHE,RUN,OUT)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':main()
