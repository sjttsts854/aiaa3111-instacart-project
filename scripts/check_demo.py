"""Integration check of the real study demo and its empty-repeat edge case."""
import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

root=Path(__file__).resolve().parents[1]
run=root/"autoresearch/research-v2"
selected=json.loads((run/"selection.json").read_text(encoding="utf-8"))["selected"]
per=pd.read_parquet(run/"confirmation"/selected/"42/users.parquet")
app=AppTest.from_file(str(root/"app.py"),default_timeout=60).run()
assert not app.exception,[x.message for x in app.exception]
assert len(app.tabs)==4
assert app.tabs[0].label=="最新补充研究"
assert len(app.selectbox[0].options)==8000
app.button[0].click().run()
assert not app.exception,[x.message for x in app.exception]
empty_user=int(per.loc[per.repeat_targets.eq(0),"user_id"].iloc[0])
app.selectbox[0].set_value(empty_user).run()
app.button[0].click().run()
assert not app.exception,[x.message for x in app.exception]
assert any("复购目标 0 种" in element.value for element in app.markdown)
print(json.dumps({"status":"PASS","tabs":4,"confirmation_users":8000,
                  "selected":selected,"empty_repeat_case":empty_user},ensure_ascii=False))
