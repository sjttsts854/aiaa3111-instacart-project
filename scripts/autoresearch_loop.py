"""Bounded classic autoresearch: commit one configuration, verify, guard, keep/revert.

The evaluator, split, labels, and experiment hypotheses are frozen before the loop.
No test labels are opened here. Only configs/candidate.json may change in the loop.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


def git(*args, check=True):
    return subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "-c", "user.name=Autoresearch", "-c",
                           "user.email=autoresearch@local.invalid", *args], text=True,
                          capture_output=True, check=check).stdout.strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/processed/10k_seed42")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    git("rev-parse", "--git-dir")
    if git("status", "--porcelain"):
        raise SystemExit("Commit your changes before running the bounded experiment loop.")
    if git("symbolic-ref", "--quiet", "HEAD",check=False)=="":
        raise SystemExit("Detached HEAD is not supported.")
    if (ROOT/".git/index.lock").exists(): raise SystemExit("Git index lock exists.")
    run = Path(args.run_dir or datetime.now().strftime("autoresearch/loop-%y%m%d-%H%M%S"))
    if run.exists(): raise SystemExit("Run directory already exists; use a new one.")
    run.mkdir(parents=True)
    plan_path = ROOT/"configs/experiments.json"
    entries = json.loads(plan_path.read_text(encoding="utf-8"))
    candidate = ROOT/"configs/candidate.json"
    frozen = {str(p.relative_to(ROOT)):digest(p) for p in
              [*ROOT.glob("basketlab/*.py"),*ROOT.glob("tests/*.py"), ROOT/"run.py",plan_path]}
    # Also freeze data-derived features and split, including the protocol manifest.
    for name in ["features.parquet","splits.csv","manifest.json","labels_train.parquet","labels_valid.parquet"]:
        p = Path(args.cache)/name; frozen[str(p)] = digest(p)
    (run/"experiment_plan.json").write_text(json.dumps(entries,indent=2),encoding="utf-8")
    (run/"frozen_files.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8")
    settings = {"skill":"uditgoenka/autoresearch", "skill_version":"2.3.0", "mode":"classic",
                "metric":"validation macro Recall@10", "direction":"higher_is_better",
                "scope":["configs/candidate.json"], "iterations":len(entries)-1,
                "forks":False, "verify":"python run.py verify --config configs/candidate.json",
                "guard":"python -m pytest -q", "cache":args.cache,
                "test_policy":"One evaluation of frozen comparisons after selection; no test feedback into tuning."}
    (run/"config.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    subprocess.run([sys.executable,"-m","pytest","-q"],check=True)
    log = run/"loop-results.tsv"
    with log.open("w",newline="",encoding="utf-8") as out:
        out.write("# metric_direction: higher_is_better\n")
        csv.writer(out,delimiter="\t").writerow(["iteration","timestamp","commit","metric","delta","guard","guard-metric","status","description"])
    incumbent = None; summaries=[]
    for i, entry in enumerate(entries):
        # Review git/log history before each isolated configuration change.
        print(git("log","--oneline","-3"),flush=True)
        if incumbent is not None: print(f"Incumbent: {incumbent['name']} {incumbent['metric']:.6f}",flush=True)
        for path, expected in frozen.items():
            if digest(ROOT/path)!=expected: raise RuntimeError(f"Frozen protocol changed: {path}")
        config = entry["config"]
        candidate.write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
        changed = bool(git("diff","--",str(candidate)))
        if changed:
            git("add","--","configs/candidate.json")
            git("commit","-m",f"experiment: {entry['name']}")
        commit = git("rev-parse","--short","HEAD")
        outdir = run/"validation"/entry["name"]
        verify_cmd = [sys.executable,"run.py","verify","--cache",args.cache,"--output",str(outdir)]
        print(f"[{i}/{len(entries)-1}] {entry['name']}: {entry['hypothesis']}",flush=True)
        result = subprocess.run(verify_cmd,text=True,capture_output=True)
        (run/f"{i:02d}-verify.log").write_text(result.stdout+result.stderr,encoding="utf-8")
        guard = subprocess.run([sys.executable,"-m","pytest","-q"],text=True,capture_output=True)
        (run/f"{i:02d}-guard.log").write_text(guard.stdout+guard.stderr,encoding="utf-8")
        metric=None;delta=0.;status="crash"
        if result.returncode==0 and guard.returncode==0:
            try: metric=float(result.stdout.strip().splitlines()[-1])
            except (ValueError,IndexError): status="metric-error"
            else:
                if not 0 <= metric <= 1: status="metric-error"
                else:
                    delta=0. if incumbent is None else metric-incumbent["metric"]
                    status="baseline" if incumbent is None else ("keep" if delta>1e-10 else "discard")
                    if status in {"baseline","keep"}:
                        incumbent={"name":entry["name"],"metric":metric,"config":config,"commit":commit}
        if status not in {"baseline","keep"} and changed:
            git("revert","HEAD","--no-edit")
        row=[i,datetime.now().isoformat(),commit,metric,delta,"pass" if guard.returncode==0 else "fail","-",status,entry["hypothesis"]]
        with log.open("a",newline="",encoding="utf-8") as out:csv.writer(out,delimiter="\t").writerow(row)
        summaries.append({"name":entry["name"],"status":status,"metric":metric,"delta":delta})
        print(f"  {status}: recall={metric}, delta={delta:+.6f}",flush=True)
        if metric is None or guard.returncode!=0:
            raise RuntimeError(f"Experiment failed. Evidence preserved in {run}.")
    (run/"selected.json").write_text(json.dumps(incumbent,indent=2),encoding="utf-8")
    (run/"validation_summary.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    handoff={"version":"2.1.0","source":"loop","timestamp":datetime.now().isoformat(),
             "status":"BOUNDED","results_tsv":str(log),"findings":summaries,"config":settings}
    (run/"handoff.json").write_text(json.dumps(handoff,indent=2),encoding="utf-8")
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/latest_run.txt").write_text(str(run.resolve()),encoding="utf-8")
    print(f"SELECTED {incumbent['name']} {incumbent['metric']:.6f}\nRUN {run}",flush=True)


if __name__=="__main__": main()
