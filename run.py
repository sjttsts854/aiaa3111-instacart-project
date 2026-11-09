"""Project entry point. Run `python run.py --help` for commands."""
import argparse
import json
import os
from pathlib import Path

# Avoid excessive BLAS threads and the known tiny-batch KMeans/MKL warning on Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "enrich", "verify", "finalize"])
    parser.add_argument("--raw", type=Path, default=Path("archive"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/10k_seed42"))
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("configs/candidate.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/verification"))
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.command=="prepare":
        from basketlab.data import prepare
        result = prepare(args.raw,args.cache,args.sample_size,args.seed)
    elif args.command=="enrich":
        from basketlab.enrichment import enrich
        result = enrich(args.cache,args.seed)
    elif args.command=="verify":
        from basketlab.experiment import verify
        result = verify(args.cache,args.output,json.loads(args.config.read_text(encoding="utf-8")))
        print(f"{result['recall_at_10']:.12f}")
        return
    else:
        if args.run is None: parser.error("--run is required for finalize")
        from basketlab.experiment import finalize
        result = finalize(args.cache,args.run)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
