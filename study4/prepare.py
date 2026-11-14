from pathlib import Path
import json
import pandas as pd
from research.data import prepare

def main():
    p=json.loads(Path('study4/protocol.json').read_text(encoding='utf-8'))
    cache=Path('data/processed/study4')
    if (cache/'manifest.json').exists():raise FileExistsError('Prepared study already exists')
    old=pd.concat([pd.read_csv(f)[['user_id']] for f in p['excluded_sources']]).drop_duplicates()
    if len(old)!=76000:raise AssertionError('Expected all 76000 previously observed users')
    old.to_csv(p['excluded_users'],index=False)
    print(prepare(p,cache),flush=True)

if __name__=='__main__':main()
