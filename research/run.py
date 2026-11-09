import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT","4")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["prepare","features","develop","freeze","confirm","report","audit"])
    p.add_argument("--protocol",type=Path,default=Path("research/protocol.json"))
    p.add_argument("--cache",type=Path,default=Path("data/processed/research_v2"))
    p.add_argument("--run",type=Path,default=Path("autoresearch/research-v2"))
    args=p.parse_args();config=json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.stage=="prepare":
        from .data import prepare
        result=prepare(config,args.cache)
    elif args.stage=="features":
        from .features import build
        result=build(config,args.cache)
    else:
        from . import experiment
        result=getattr(experiment,args.stage)(config,args.cache,args.run)
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str))


if __name__=="__main__":main()
