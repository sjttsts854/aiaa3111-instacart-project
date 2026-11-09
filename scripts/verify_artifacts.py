"""Independent acceptance check of saved artifacts; does not fit or rank models."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
run = Path((ROOT/"outputs/latest_run.txt").read_text(encoding="utf-8").strip())
config = json.loads((run/"config.json").read_text(encoding="utf-8"))
cache = ROOT/config["cache"]
frozen = json.loads((run/"frozen_files.json").read_text(encoding="utf-8"))
for name, expected in frozen.items():
    actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    assert actual==expected, f"Frozen protocol changed: {name}"
splits=pd.read_csv(cache/"splits.csv")
assert not splits.user_id.duplicated().any()
features=pd.read_parquet(cache/"features.parquet")
assert "label" not in features and not features.duplicated(["user_id","product_id"]).any()
selected=json.loads((run/"selected.json").read_text(encoding="utf-8"))
validation=json.loads((run/"validation_summary.json").read_text(encoding="utf-8"))
assert selected["metric"]==max(x["metric"] for x in validation)
test=features[features.split.eq("test")]
truth=pd.read_parquet(cache/"labels_test.parquet")
top=pd.read_parquet(run/"test/selected_predictions.parquet")
candidate_sets=test.groupby("user_id").product_id.agg(set).to_dict()
truth_sets=truth.groupby("user_id").product_id.agg(set).to_dict()
pred_sets=top.groupby("user_id").product_id.agg(set).to_dict()
recalls=[];empty=0
for user,candidates in candidate_sets.items():
    target=candidates & truth_sets[user]
    predicted=pred_sets[user]
    assert predicted<=candidates
    assert len(predicted)==min(10,len(candidates))
    if target:recalls.append(len(predicted & target)/len(target))
    else:empty+=1
metrics=pd.read_csv(run/"test/metrics.csv")
row=metrics[metrics.name.eq(selected["name"])].iloc[0]
assert np.isclose(np.mean(recalls),row.recall_at_10,atol=1e-12)
assert row.empty_repeat_targets==empty
assert set(candidate_sets)==set(splits.loc[splits.split.eq("test"),"user_id"])
completed=json.loads((run/"test/completed.json").read_text(encoding="utf-8"))
assert hashlib.sha256((run/"test/metrics.csv").read_bytes()).hexdigest()==completed["metrics_sha256"]
print(json.dumps({"status":"PASS","frozen_files":len(frozen),"test_users":len(candidate_sets),
                  "nonempty_repeat_targets":len(recalls),"empty_repeat_targets":empty,
                  "independent_recall_at_10":float(np.mean(recalls)),"selected":selected["name"]},indent=2))
