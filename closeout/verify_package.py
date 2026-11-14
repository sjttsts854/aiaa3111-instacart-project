"""Independently validate the portable frozen evidence, without training."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root=ROOT):
    manifest=json.loads((root/'package_manifest.json').read_text(encoding='utf-8'))
    for name,value in manifest['files'].items():
        path=root/name
        if not path.is_file() or digest(path)!=value['sha256']:raise AssertionError(f'Package integrity mismatch: {name}')
    d=root/'data_demo';users=pd.read_parquet(d/'users.parquet');catalog=pd.read_parquet(d/'historical_catalog.parquet')
    targets=pd.read_parquet(d/'targets.parquet');preds=pd.read_parquet(d/'predictions.parquet');products=pd.read_parquet(d/'products.parquet')
    assert users.user_id.nunique()==len(users)==8000
    assert not catalog.duplicated(['user_id','product_id']).any()
    assert not targets.duplicated(['user_id','product_id']).any()
    assert not preds.duplicated(['policy','user_id','product_id']).any()
    keys=set(zip(catalog.user_id,catalog.product_id));truth=set(zip(targets.user_id,targets.product_id))
    np.testing.assert_array_equal(targets.is_repeat,[pair in keys for pair in zip(targets.user_id,targets.product_id)])
    assert set(targets.product_id)|set(catalog.product_id)<=set(products.product_id)
    repeats=targets[targets.is_repeat].groupby('user_id').size().reindex(users.user_id,fill_value=0)
    assert repeats.gt(0).sum()==7472 and repeats.eq(0).sum()==528
    assert set(preds.user_id)==set(users.user_id)
    budget=catalog.groupby('user_id')[['dual_eligible','recency_eligible']].sum()
    np.testing.assert_array_equal(budget.dual_eligible,budget.recency_eligible)
    observed=catalog[catalog.orders_since_purchase.lt(5)]
    assert observed.dual_eligible.all() and observed.recency_eligible.all()
    rows=[]
    for policy in ['full','dual','recency_budget']:
        sub=preds[preds.policy.eq(policy)]
        allowed=catalog if policy=='full' else catalog[catalog['dual_eligible' if policy=='dual' else 'recency_eligible']]
        assert set(zip(sub.user_id,sub.product_id))<=set(zip(allowed.user_id,allowed.product_id))
        np.testing.assert_array_equal(sub.label,[int(pair in truth) for pair in zip(sub.user_id,sub.product_id)])
        assert sub.groupby('user_id').size().max()<=10
        hits=sub.groupby('user_id').label.sum().reindex(users.user_id,fill_value=0)
        n=sub.groupby('user_id').size().reindex(users.user_id,fill_value=0)
        recall=float((hits/repeats.replace(0,np.nan)).mean())
        expected=json.loads((d/'seed42_metrics.json').read_text())[policy]
        np.testing.assert_allclose(recall,expected['recall_at_10'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(float(n.mean()),expected['mean_recommendation_length'],atol=1e-12,rtol=0)
        rows.append({'policy':policy,'single_seed_recall_at_10':recall})
    return {'status':'PASS','files_checked':len(manifest['files']),'users':8000,'nonempty_targets':7472,
            'scope':'Portable artifact integrity and full-target seed42 case metrics; not a raw-source or retraining audit','metrics':rows}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
