from pathlib import Path
import json
import pandas as pd
from research.data import prepare
from research.patterns import pair_rules,score_rules

def main():
    config=json.loads(Path('study3/protocol.json').read_text(encoding='utf-8'))
    out=Path('data/processed/study3')
    if not (out/'manifest.json').exists():
        old=pd.concat([pd.read_csv(p)[['user_id']] for p in config['excluded_sources']]).drop_duplicates()
        old.to_csv(config['excluded_users'],index=False)
        print(prepare(config,out),flush=True)
    if not (out/'features.parquet').exists():
        c=pd.read_parquet(out/'candidates.parquet');h=pd.read_parquet(out/'history.parquet')
        users=pd.read_csv(out/'splits.csv');train=users.loc[users.split.eq('train'),'user_id']
        rules,info=pair_rules(h,train,**config['rule_settings'])
        f=score_rules(h,c,rules,'co_narrow')
        c.merge(f,on=['user_id','product_id'],validate='one_to_one').to_parquet(out/'features.parquet',index=False)
        rules.to_parquet(out/'rules.parquet',index=False)
        (out/'rules_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
        print(info,flush=True)

if __name__=='__main__':main()
