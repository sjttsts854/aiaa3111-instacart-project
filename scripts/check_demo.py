"""Integration check using real completed study artifacts, including empty targets.

Use --require-study4 for the final check after fourth-study report generation.
Without it, a pending fourth study checks the explicit waiting state only.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

parser=argparse.ArgumentParser()
parser.add_argument("--require-study4",action="store_true")
parser.add_argument("--require-followup",action="store_true")
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
latest=root/"autoresearch/research-v4"
confirmation=latest/"confirmation"
study4_ready=(confirmation/"completed.json").exists()
report=root/"reports/study4"
followup=root/"autoresearch/research-v4-followup"
followup_report=root/"reports/study4_followup"
followup_ready=(followup/"completed.json").exists()
if args.require_study4 or args.require_followup:
    assert study4_ready,"Fourth-study confirmation is incomplete"
    assert (report/"research_story.md").exists(),"Fourth-study report is missing"
if args.require_followup:
    assert followup_ready,"Posthoc followup is incomplete"
    assert (followup_report/"analysis.md").exists(),"Posthoc report is missing"
run=root/"autoresearch/research-v2"
selected=json.loads((run/"selection.json").read_text(encoding="utf-8"))["selected"]
per=pd.read_parquet(run/"confirmation"/selected/"42/users.parquet")
app=AppTest.from_file(str(root/"app.py"),default_timeout=60).run()
assert not app.exception,[x.message for x in app.exception]
assert len(app.tabs)==5
assert [tab.label for tab in app.tabs]==["购物记忆研究","第三轮补充研究","购物清单演示（第二轮）","第二轮实验结果","第二轮研究发现"]
memory=app.tabs[0]
if study4_ready:
    metrics=pd.read_csv(confirmation/"metrics.csv").set_index("name")
    contrast=pd.read_csv(confirmation/"primary_contrasts.csv").set_index("contrast").loc["dual_vs_full"]
    protocol=json.loads((latest/"freeze.json").read_text(encoding="utf-8"))["protocol"]
    shown_metrics={metric.label:metric.value for metric in memory.metric}
    for name,label in [("full","完整历史候选"),("recent","只留最近 5 单"),("dual","验证选定策略")]:
        assert shown_metrics[f"{label} · Recall@10"]==f"{metrics.loc[name,'recall_at_10']:.2%}"
    conclusion="通过" if contrast.sim_low>=-protocol["practical_margin"] else "未通过"
    assert any(f"非劣结论：{conclusion}预先设定" in element.value for element in memory.markdown)
    assert any(f"全体用户候选总量减少 {1-metrics.loc['dual','mean_candidates']/metrics.loc['full','mean_candidates']:.1%}" in element.value for element in memory.markdown)
    primary=pd.read_csv(confirmation/"primary_contrasts.csv").set_index("contrast")
    for name,label in [("dual_vs_full","完整历史候选"),("dual_vs_frequency_budget","同预算频次对照"),("dual_vs_recency_budget","同预算最近购买对照")]:
        row=primary.loc[name]
        expected="同时区间支持选定策略的召回率更高" if row.sim_low>0 else ("同时区间支持选定策略的召回率更低" if row.sim_high<0 else "同时区间包含零，未证实选定策略的召回率更高")
        assert any(f"相对{label}，差异 {100*row.delta:+.4f} 个百分点；{expected}" in element.value for element in memory.markdown)
    observed=pd.read_csv(confirmation/"history_observed/sensitivity.csv")
    shown_observed=next(element.value for element in memory.dataframe if "全体原始差异（百分点）" in element.value.columns)
    assert len(shown_observed)==4
    for original,shown in [("raw_overall_long_minus_short","全体原始差异（百分点）"),
                           ("raw_common_support_long_minus_short","同支持原始差异（百分点）"),
                           ("standardized_long_minus_short","同支持标准化差异（百分点）")]:
        pd.testing.assert_series_equal(shown_observed[shown],(100*observed[original]).round(3),check_names=False)
    shown_policies=next(element.value for element in memory.dataframe if "人均候选数" in element.value.columns)
    assert len(shown_policies)==5
    assert shown_policies.loc["完整历史候选","人均候选数"]==metrics.loc["full","mean_candidates"]
    assert shown_policies.loc["验证选定策略","Recall@10"]==metrics.loc["dual","recall_at_10"]
    grids=[element.value for element in memory.dataframe if list(element.value.columns)==["最近 5 单候选","完整历史候选"]]
    assert len(grids)==2 and all(grid.shape==(2,2) for grid in grids)
    factorial=pd.read_csv(confirmation/"factorial_metrics.csv",dtype={"score_window":str,"pool":str})
    factorial=factorial[factorial.subgroup.eq("long_history") & factorial.score_window.isin(["5","all"]) & factorial.pool.isin(["5","all"])]
    for grid,method in zip(grids,["frequency","decay"]):
        expected=factorial[factorial.method.eq(method)].pivot(index="score_window",columns="pool",values="recall_at_10")
        expected=(100*expected.reindex(index=["5","all"],columns=["5","all"])).round(3)
        expected.index=["只用最近 5 单评分","用全部历史评分"]
        expected.columns=["最近 5 单候选","完整历史候选"]
        pd.testing.assert_frame_equal(grid,expected)
    ledger=next(element.value for element in memory.dataframe if "净变化（百分点）" in element.value.columns)
    assert len(ledger)==4
    evidence=pd.read_csv(confirmation/"ledger_summary.csv")
    evidence=evidence[evidence.subgroup.eq("long_history")].copy()
    evidence["method"]=evidence.name.str.replace(r"_\d+$","",regex=True)
    evidence=evidence.groupby("method")[["new_true_recall","lost_old_true_recall","delta_recall"]].mean()
    evidence.index=evidence.index.to_series().replace({"frequency":"累计频次 · 扩展到完整历史","decay":"近期加权 · 扩展到完整历史","learned_full":"学习模型 · 扩展到完整历史","learned_dual":"学习模型 · 扩展到选定策略"})
    evidence=(100*evidence).round(3).rename(columns={"new_true_recall":"新增命中贡献（百分点）","lost_old_true_recall":"挤出命中损失（百分点）","delta_recall":"净变化（百分点）"})
    pd.testing.assert_frame_equal(ledger,evidence)
    if args.require_study4 or args.require_followup:
        for name in ["01_observed_history.png","02_factorial_memory.png","03_primary_contrasts.png","04_expansion_ledger.png","05_budget_tradeoff.png"]:
            assert (report/"figures"/name).is_file(),f"Missing fourth-study figure: {name}"
        assert any(button.label=="下载购物记忆研究证据稿" for button in memory.get("download_button"))
    if followup_ready:
        changes=pd.read_csv(followup/"users/change_summary.csv").set_index("comparator")
        overlap=pd.read_csv(followup/"users/overlap_summary.csv").set_index(["population","comparator"])
        row=changes.loc["full"]
        assert any("没有新增训练、重选参数或启用新的确认用户" in element.value for element in memory.info)
        assert any(f"{int(row.improved_users):,} 人改善、{int(row.harmed_users):,} 人受损、{int(row.unchanged_users):,} 人不变" in element.value for element in memory.markdown)
        assert shown_metrics["Recall 不变 · 非空目标用户"]==f"{row.unchanged_fraction:.2%}"
        assert shown_metrics["与完整候选同清单 · 三模型均一致"]==f"{overlap.loc[('all','full'),'top_all_seeds_ordered_identical_fraction']:.2%}"
        assert shown_metrics["与同预算近期策略同清单 · 三模型均一致"]==f"{overlap.loc[('all','recency_budget'),'top_all_seeds_ordered_identical_fraction']:.2%}"
        changed_table=next(element.value for element in memory.dataframe if "改善人数" in element.value.columns)
        assert changed_table.loc["完整候选","有复购目标用户"]==row.nonempty_users
        assert changed_table.loc["完整候选","受损人数"]==row.harmed_users
        assert changed_table.loc["完整候选","平均变化（百分点）"]==round(100*row.mean_delta,4)
        depth=pd.read_csv(followup/"depth/contrasts.csv")
        expected=depth.loc[depth.comparator.eq("full")].set_index("k")
        expected=(100*expected[["delta","sim_low","sim_high"]]).round(4).rename(columns={"delta":"事后 Recall 差异（百分点）","sim_low":"事后同时区间下界","sim_high":"事后同时区间上界"})
        expected.index.name="最多推荐件数 K"
        shown_depth=next(element.value for element in memory.dataframe if "事后 Recall 差异（百分点）" in element.value.columns)
        pd.testing.assert_frame_equal(shown_depth,expected)
        assert any("不能仅凭 K=20 的点估计损失就宣布原定非劣检验失败" in element.value for element in memory.caption)
        if args.require_followup:
            for name in ["01_user_changes.png","02_historical_subgroups.png","03_list_depth.png","04_target_ranks.png"]:
                assert (followup_report/"figures"/name).is_file(),f"Missing posthoc figure: {name}"
            assert any(button.label=="下载事后分析报告" for button in memory.get("download_button"))
else:
    assert any("第四轮独立确认尚未完成" in element.value for element in memory.info)
    assert not memory.metric
assert any("第二轮（V2）" in element.value for element in app.tabs[2].caption)
assert len(app.selectbox[0].options)==8000
next(button for button in app.button if button.label=="揭晓真实下一单").click().run()
assert not app.exception,[x.message for x in app.exception]
empty_user=int(per.loc[per.repeat_targets.eq(0),"user_id"].iloc[0])
app.selectbox[0].set_value(empty_user).run()
next(button for button in app.button if button.label=="揭晓真实下一单").click().run()
assert not app.exception,[x.message for x in app.exception]
assert any("复购目标 0 种" in element.value for element in app.markdown)
print(json.dumps({"status":"PASS","tabs":5,"case_study":"V2","case_confirmation_users":8000,
                  "selected":selected,"empty_repeat_case":empty_user,
                  "study4_confirmation":"complete" if study4_ready else "pending",
                  "study4_result_branch_tested":study4_ready,
                  "study4_report_required":args.require_study4,
                  "posthoc_followup_branch_tested":followup_ready,
                  "posthoc_report_required":args.require_followup},ensure_ascii=False))
