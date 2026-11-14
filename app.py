"""Local, read-only demonstration using frozen held-out predictions."""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Next Basket Lab", page_icon="🛒", layout="wide")
st.title("下一单复购清单预测")
st.caption("研究哪些历史信息有助于排序，以及保留哪些历史商品作为下一单候选。")
study_run = ROOT / "autoresearch/research-v2"
is_study = (study_run/"confirmation/completed.json").exists()
if is_study:
    run = study_run
    cache = ROOT / "data/processed/research_v2"
    selected = {"name":json.loads((run/"selection.json").read_text(encoding="utf-8"))["selected"]}
else:
    pointer = ROOT / "outputs/latest_run.txt"
    if not pointer.exists():
        st.info("请先运行数据准备、实验和最终确认。操作说明见项目 README。")
        st.stop()
    run = Path(pointer.read_text(encoding="utf-8").strip())
    if not (run/"test/completed.json").exists():
        st.info("实验尚未完成最终测试。完成后这里会自动展示结果。")
        st.stop()
    config = json.loads((run/"config.json").read_text(encoding="utf-8"))
    cache = ROOT / config["cache"]
    selected = json.loads((run/"selected.json").read_text(encoding="utf-8"))


@st.cache_data
def load_tables(cache_path, run_path, study, selected_name):
    c, r = Path(cache_path), Path(run_path)
    part="confirmation" if study else "test"
    users=pd.read_parquet(c/"users.parquet")
    users=users[users.split.eq("confirm" if study else "test")]
    history=pd.read_parquet(c/"history.parquet")
    history=history[history.user_id.isin(users.user_id)]
    return {
        "users": users,
        "history": history,
        "products": pd.read_parquet(c/"products.parquet"),
        "rules": pd.read_parquet(c/"rules_co_wide.parquet") if study else pd.read_csv(c/"rules.csv"),
        "predictions": pd.read_parquet(r/part/selected_name/"42/top10.parquet" if study else r/"test/selected_predictions.parquet"),
        "metrics": pd.read_csv(r/part/"metrics.csv"),
        "per_user": pd.read_parquet(r/part/selected_name/"42/users.parquet") if study else pd.read_csv(r/f"test/{selected_name}_users.csv"),
    }


def show_memory_study():
    """Render completed fourth-study evidence without presupposing pruning helps."""
    latest = ROOT / "autoresearch/research-v4"
    confirmation = latest / "confirmation"
    report_dir = ROOT / "reports/study4"
    st.subheader("订单越多，下一单为何可能更难预测？应该保留哪些购物记忆？")
    st.write("把两个问题拆开检验：排序时看多久的历史，以及允许哪些历史商品进入候选清单。")
    if not (confirmation / "completed.json").exists():
        st.info("第四轮独立确认尚未完成。完成并生成证据稿后，这里会自动展示实际结果；第三轮与第二轮结果仍保留在后续标签页。")
        st.caption("实验预先固定目标分母与同一批用户；删除候选商品不会删除评价中的真实目标。")
        return

    metrics = pd.read_csv(confirmation / "metrics.csv").set_index("name")
    primary = pd.read_csv(confirmation / "primary_contrasts.csv").set_index("contrast")
    freeze = json.loads((latest / "freeze.json").read_text(encoding="utf-8"))
    protocol = freeze["protocol"]
    policy = freeze["selection"]["policy"]
    margin = float(protocol["practical_margin"])
    full = metrics.loc["full"]
    st.write(f"第四轮使用新的 {int(full['users']):,} 名确认用户；训练、验证与候选策略选择在查看本轮确认结果前完成。")
    names = {"full": "完整历史候选", "recent": "只留最近 5 单", "dual": "验证选定策略",
             "frequency_budget": "同预算 · 按累计频次", "recency_budget": "同预算 · 按最近购买"}
    for column, name in zip(st.columns(3), ["full", "recent", "dual"]):
        column.metric(f"{names[name]} · Recall@10", f"{metrics.loc[name, 'recall_at_10']:.2%}")
        column.caption(f"人均候选 {metrics.loc[name, 'mean_candidates']:.2f} 件")
    if policy["kind"] == "dual":
        st.caption(f"选定策略：最近 {policy['window']} 单商品，加上至多 {policy['m']} 件更早的常购商品。所有策略使用同一完整历史模型的固定排序分数。")
    else:
        st.caption("验证阶段未选出符合条件的候选缩减策略，按预先规则回退到完整历史候选。")
    st.caption("Recall@10：最多推荐 10 件，对下一单有复购目标的用户求平均；三个训练种子的用户指标先取均值。真实目标始终包含完整历史范围内的下一单复购商品，被剔除的目标仍计为未命中。")

    def show_figure(filename):
        path = report_dir / "figures" / filename
        if path.exists():
            st.image(str(path))

    def signed_pp(value):
        return f"{100 * value:+.2f} 个百分点"

    st.subheader("A · 历史长短差异：先区分筛选用户与重新加权")
    sensitivity = pd.read_csv(confirmation / "history_observed/sensitivity.csv")
    displayed = sensitivity[["specification", "raw_overall_long_minus_short",
                             "raw_common_support_long_minus_short", "standardized_long_minus_short",
                             "standardized_ci95_low", "standardized_ci95_high", "retained_users"]].copy()
    for field in displayed.columns[1:-1]:
        displayed[field] = (100 * displayed[field]).round(3)
    displayed = displayed.rename(columns={
        "specification": "预设分层", "raw_overall_long_minus_short": "全体原始差异（百分点）",
        "raw_common_support_long_minus_short": "同支持原始差异（百分点）",
        "standardized_long_minus_short": "同支持标准化差异（百分点）",
        "standardized_ci95_low": "标准化 95% 下界", "standardized_ci95_high": "标准化 95% 上界",
        "retained_users": "保留用户数"})
    st.dataframe(displayed, hide_index=True)
    st.caption("差异均为 16+ 单减 3–5 单。四种预设设置只按历史候选数，或历史候选数 × 历史平均每单商品数分层，每组每格至少 10 或 20 人。相同支持内，三个历史长度组使用相同的总体分层权重。")
    evaluable = sensitivity.standardized_long_minus_short.dropna()
    if len(evaluable):
        st.write(f"全体原始差异为 {signed_pp(sensitivity.iloc[0].raw_overall_long_minus_short)}；可计算的 {len(evaluable)} 套历史分层标准化差异范围为 {signed_pp(evaluable.min())} 至 {signed_pp(evaluable.max())}。方向与数值均以表中实际结果为准。")
    else:
        st.info("四套设置都缺少足够的共同支持，不能据此给出标准化差异。")
    cells = pd.read_csv(confirmation / "history_observed/cells.csv")
    identical_threshold_support = []
    for binning, specifications in sensitivity.groupby("binning"):
        blocks = [cells.loc[cells.specification.eq(name)].drop(columns="specification")
                  .sort_values(["history_group", "candidate_bin", "historical_basket_bin"]).reset_index(drop=True)
                  for name in specifications.specification]
        if len(blocks) > 1 and not blocks[0].empty and all(blocks[0].equals(block) for block in blocks[1:]):
            identical_threshold_support.append(binning)
    if len(identical_threshold_support) == 2:
        st.info("本轮每种分层下，最低人数 10 与 20 产生完全相同的支持集合、权重和结果。因此四套预设检查实际只有两种标准化结果，不能算作四次独立重复。")
    with st.expander("查看每个历史组保留了多少用户"):
        retention = pd.read_csv(confirmation / "history_observed/groups.csv")
        st.dataframe(retention, hide_index=True)
    st.caption("全体→同支持会改变样本；同支持→标准化才是这批用户的重新加权。虽然分层不用未来目标大小，Recall 仍要求未来复购目标非空；这不是历史长度的因果效应。区间以已观察的分层、权重和训练模型为条件。")
    show_figure("01_observed_history.png")

    st.subheader("B · 同一批长历史用户：排序信息 × 候选范围")
    factorial = pd.read_csv(confirmation / "factorial_metrics.csv", dtype={"score_window": str, "pool": str})
    core = factorial[factorial.subgroup.eq("long_history") & factorial.score_window.isin(["5", "all"]) & factorial.pool.isin(["5", "all"])]
    for column, method, title in zip(st.columns(2), ["frequency", "decay"], ["累计频次排序", "近期加权排序"]):
        grid = core[core.method.eq(method)].pivot(index="score_window", columns="pool", values="recall_at_10")
        grid = (100 * grid.reindex(index=["5", "all"], columns=["5", "all"])).round(3)
        grid.index = ["只用最近 5 单评分", "用全部历史评分"]
        grid.columns = ["最近 5 单候选", "完整历史候选"]
        column.write(title + " · Recall@10（%）")
        column.dataframe(grid)
    st.caption("主比较固定历史至少 16 单且复购目标非空的同一批用户。四格都知道完整历史商品目录，改变的是评分所用订单与允许进入排名的候选；没有删掉评价目标。")
    st.write(f"固定完整候选后，把评分信息从最近 5 单扩到全部历史：累计频次变化 {signed_pp(primary.loc['frequency_information', 'delta'])}，近期加权变化 {signed_pp(primary.loc['decay_information', 'delta'])}。固定全部历史评分后，扩展候选范围的变化分别为 {signed_pp(primary.loc['frequency_expansion', 'delta'])} 与 {signed_pp(primary.loc['decay_expansion', 'delta'])}。旧历史的净作用随排序规则改变。")
    factorial_contrasts = primary.loc[[f"{method}_{effect}" for method in ["frequency", "decay"] for effect in ["information", "expansion", "interaction"]], ["delta", "sim_low", "sim_high"]]
    with st.expander("查看评分信息、候选扩展与交互效应的区间"):
        st.dataframe((100 * factorial_contrasts).round(3).rename(columns={"delta": "差异（百分点）", "sim_low": "同时 95% 下界", "sim_high": "同时 95% 上界"}))
        st.caption("information：完整候选中，全部历史评分减近期评分；expansion：全部历史评分下，完整候选减近期候选；interaction：两种改变的差中之差。")
    show_figure("02_factorial_memory.png")
    ledgers = pd.read_csv(confirmation / "ledger_summary.csv")
    ledgers = ledgers.loc[ledgers.subgroup.eq("long_history")].copy()
    ledgers["method"] = ledgers.name.str.replace(r"_\d+$", "", regex=True)
    ledger = ledgers.groupby("method")[["new_true_recall", "lost_old_true_recall", "delta_recall"]].mean()
    ledger.index = ledger.index.to_series().replace({"frequency": "累计频次 · 扩展到完整历史", "decay": "近期加权 · 扩展到完整历史",
                                                   "learned_full": "学习模型 · 扩展到完整历史", "learned_dual": "学习模型 · 扩展到选定策略"})
    st.write("扩大候选范围的收益账本：新增真实命中 − 被挤出的原有真实命中 = 净召回变化。")
    st.dataframe((100 * ledger).round(3).rename(columns={"new_true_recall": "新增命中贡献（百分点）", "lost_old_true_recall": "挤出命中损失（百分点）", "delta_recall": "净变化（百分点）"}))
    st.caption("从最近 5 单候选出发，扩大候选时保持分数和排序规则不变；这里显示长历史用户的平均召回贡献，学习模型取三个种子均值。账本是精确命中分解，不是消费者行为的因果解释。")
    show_figure("04_expansion_ledger.png")

    st.subheader("C · 少保留一些候选，代价与收益是多少？")
    policies = metrics.loc[list(names), ["recall_at_10", "mean_candidates", "candidate_coverage"]].copy()
    policies.index = policies.index.to_series().map(names)
    policies = policies.rename(columns={"recall_at_10": "Recall@10", "mean_candidates": "人均候选数", "candidate_coverage": "复购目标覆盖率"})
    st.dataframe(policies.style.format({"Recall@10": "{:.2%}", "人均候选数": "{:.2f}", "复购目标覆盖率": "{:.2%}"}))
    comparison = primary.loc["dual_vs_full"]
    retained = metrics.loc["dual", "mean_candidates"] / full["mean_candidates"]
    ni_pass = bool(comparison.sim_low >= -margin)
    st.write(f"选定策略使全体用户候选总量减少 {1 - retained:.1%}（保留 {retained:.1%}）；相对完整候选的召回差异为 {100 * comparison.delta:+.4f} 个百分点，同时 95% 区间为 [{100 * comparison.sim_low:+.4f}, {100 * comparison.sim_high:+.4f}] 个百分点。")
    conclusion = "通过" if ni_pass else "未通过"
    st.write(f"非劣结论：{conclusion}预先设定的 {100 * margin:.1f} 个百分点容忍界限（比较同时区间下界）。这不等于无损，也不自动意味着提高召回率。")
    if policy["kind"] == "all":
        st.info("本轮选定策略等同完整候选，因此其非劣结果不构成候选缩减成功的证据。")
    def superiority_text(row, control):
        if row.sim_low > 0:
            inference = "同时区间支持选定策略的召回率更高"
        elif row.sim_high < 0:
            inference = "同时区间支持选定策略的召回率更低"
        else:
            inference = "同时区间包含零，未证实选定策略的召回率更高"
        return f"相对{control}，差异 {100 * row.delta:+.4f} 个百分点；{inference}。"
    st.write(superiority_text(comparison, "完整历史候选"))
    budget_rows = primary.loc[["dual_vs_frequency_budget", "dual_vs_recency_budget"], ["delta", "sim_low", "sim_high"]].copy()
    budget_rows.index = ["选定策略 − 同预算频次", "选定策略 − 同预算最近购买"]
    st.dataframe((100 * budget_rows).round(3).rename(columns={"delta": "差异（百分点）", "sim_low": "同时 95% 下界", "sim_high": "同时 95% 上界"}))
    st.write(superiority_text(primary.loc["dual_vs_frequency_budget"], "同预算频次对照"))
    st.write(superiority_text(primary.loc["dual_vs_recency_budget"], "同预算最近购买对照"))
    st.caption("同预算对照为每个用户保留与选定策略相同数量的候选。主要比较的区间对预先固定的 9 项比较作同时校正；候选数量减少本身不是实际运行提速证据。")
    show_figure("03_primary_contrasts.png")
    show_figure("05_budget_tradeoff.png")
    st.caption("这些离线结果只评价历史买过商品的下一单复购，不评价首次购买的新品，也不表示未购买就是忘买。")
    story_path = report_dir / "research_story.md"
    if story_path.exists():
        story = story_path.read_text(encoding="utf-8")
        with st.expander("阅读完整研究故事与证据"):
            st.markdown("\n".join(line for line in story.splitlines() if not line.lstrip().startswith("![")))
        st.download_button("下载购物记忆研究证据稿", story, file_name="research_story.md", mime="text/markdown")
    else:
        st.info("独立确认已完成，完整证据稿和研究图正在生成。上方数值直接来自已完成的确认结果。")


data = load_tables(str(cache),str(run),is_study,selected["name"])
tabs = st.tabs(["购物记忆研究", "第三轮补充研究", "购物清单演示（第二轮）", "第二轮实验结果", "第二轮研究发现"])
with tabs[0]:
    show_memory_study()
with tabs[1]:
    extension=ROOT/"reports/study3"
    if (extension/"research_extension.md").exists():
        latest=ROOT/"autoresearch/research-v3"
        m=pd.read_csv(latest/"confirmation/metrics.csv").set_index('name')
        st.subheader("群体信息是否有用，取决于怎样使用")
        st.write("第三轮排除此前 43,000 名用户，再抽取 33,000 名用户：20,000 训练、5,000 验证、8,000 独立确认。方法与比较在确认前固定。")
        a,b,c=st.columns(3)
        a.metric("时间频次 · 个人历史",f"{m.loc['tifu_self','recall_at_10']:.2%}")
        b.metric("TIFU-KNN · 加入邻居",f"{m.loc['tifu','recall_at_10']:.2%}")
        c.metric("完整历史特征模型",f"{m.loc['full','recall_at_10']:.2%}")
        st.caption("均为最多推荐 10 件的复购召回率。TIFU 的个人历史对照使用相同时间权重；学习模型显示三个种子的指标均值。")
        st.image(str(extension/"figures/primary_contrasts.png"))
        st.write("邻居信息相对 TIFU 的个人历史对照产生小幅增益；共购规则相对较强的完整模型未取得收益。两者使用的个人基线不同，不能只把差异归因于表示方式，也不能说所有群体信息都无效。")
        st.image(str(extension/"figures/history_sensitivity.png"))
        st.write("六套分组设置都检查了三种比较：全体用户、规模重叠的同一批用户、这批用户统一权重后。第三轮分层同时使用历史候选数和未来复购目标数，属于回顾性敏感性诊断；第四轮另行检验只用历史变量的分层。排除用户与重新加权都改变结果，不能解释为历史长度的因果效应。")
        st.warning("需要修正第二轮解释：规则覆盖内的局部收益未在新用户中稳定复现。")
        st.image(str(extension/"figures/quota_diagnostic.png"))
        st.write("为减少规则挤占推荐名额，尝试保留原模型的覆盖名额再排序。验证集选择不启用该改动，因此不能宣称新方法提高了预测表现。")
        st.download_button("下载补充研究证据稿",(extension/"research_extension.md").read_text(encoding='utf-8'),file_name='research_extension.md',mime='text/markdown')
    else:
        st.info("补充研究证据稿尚未生成；已完成的第二轮研究在其他标签页。")
with tabs[4]:
    if (ROOT/"reports/research_findings.md").exists() and is_study:
        scope=json.loads((run/"confirmation/scope.json").read_text(encoding="utf-8"))
        metrics=data["metrics"].set_index("name")
        st.subheader("从比较分数，到检验哪些信息有用")
        st.write("第一轮用于发现问题；本轮排除此前全部用户，再用 8,000 名新用户作最终确认。训练和方法选择提前完成。")
        a,b,c=st.columns(3)
        a.metric("累计频次 · 复购召回率",f"{metrics.loc['frequency','recall_at_10']:.2%}")
        b.metric("近期加权 · 复购召回率",f"{metrics.loc['decay','recall_at_10']:.2%}")
        c.metric("完整个人历史 · 复购召回率",f"{metrics.loc['full','recall_at_10']:.2%}")
        st.caption("推荐最多 10 件；学习模型取三个训练种子的指标均值，不是集成预测。只计算下一单确有复购商品的用户。")
        st.image(str(ROOT/"reports/figures/primary_contrasts.png"))
        st.write("图中比较同一批用户。绿色区间表示不确定性；虚线是预先设定的 0.5 个百分点实际意义阈值。研究结论需要同时看均值、区间和任务范围。")
        st.image(str(ROOT/"reports/figures/history_difficulty.png"))
        st.write("不同历史长度的人有不同的候选规模。统一候选数和目标数的分布后再比较，结论仅适用于这些规模重叠的人群，不是历史长度的因果效应。")
        st.info(f"复购召回率不等于整篮预测准确率。完整模型对整张下一单的召回率为 {scope['full_model_whole_basket_recall_at_10']:.2%}；新品不在当前候选范围内。")
        st.download_button("下载研究证据稿",(ROOT/"reports/research_findings.md").read_text(encoding="utf-8"),file_name="research_findings.md",mime="text/markdown")
    else:
        st.write("第一轮已验证基础流程。更完整的研究分析尚未生成。")
with tabs[2]:
    st.caption("本页展示第二轮（V2）验证选定模型的购物清单案例，不是第四轮购物记忆策略的预测。最新购物记忆研究见第一个标签页。")
    ids = data["per_user"].user_id.sort_values().astype(int).tolist()
    user = st.selectbox("选择留出的测试用户",ids)
    profile = data["users"].set_index("user_id").loc[user]
    a,b,c = st.columns(3)
    a.metric("已观察订单",int(profile.history_orders))
    b.metric("平均每单商品数",f"{profile.basket_mean:.1f}")
    c.metric("历史重复购买比例",f"{profile.historical_repeat_rate:.1%}")
    names = data["products"][["product_id","product_name"]]
    history = data["history"][data["history"].user_id.eq(user)]
    left,right = st.columns(2)
    with left:
        st.subheader("最近三单")
        for order_number in sorted(history.order_number.unique())[-3:][::-1]:
            items = history[history.order_number.eq(order_number)].merge(names,on="product_id")
            with st.expander(f"第 {order_number} 单 · {len(items)} 种商品",expanded=True):
                st.write("、".join(items.sort_values("add_to_cart_order").product_name))
    with right:
        st.subheader("下一单可能再次购买")
        top = data["predictions"][data["predictions"].user_id.eq(user)].merge(names,on="product_id")
        count = history.groupby("product_id").size().rename("历史购买次数")
        shown = top.join(count,on="product_id")[["rank","product_name","历史购买次数","score"]]
        st.dataframe(shown.rename(columns={"rank":"排序","product_name":"商品","score":"模型排序分数"}),hide_index=True)
        st.caption(f"验证集选定模型：{selected['name']}。"+("此处展示训练种子 42 的单个模型。" if is_study else "")+"分数用于排序，不宣称已校准为概率。")
        rules = data["rules"]
        last = history.loc[history.age.eq(0),"product_id"]
        matched = rules[rules.antecedent.isin(last) & rules.consequent.isin(top.product_id)].nlargest(5,"confidence")
        if len(matched):
            st.write("相关历史共购规则（不代表因果，也不一定是模型采用的理由）：")
            mapping=names.set_index("product_id").product_name
            for row in matched.itertuples():
                st.write(f"{mapping.get(row.antecedent)} → {mapping.get(row.consequent)}；置信度 {row.confidence:.1%}，提升度 {row.lift:.2f}")
    if st.button("揭晓真实下一单"):
        labels = pd.read_parquet(cache/("labels_confirm.parquet" if is_study else "labels_test.parquet"))
        actual = labels[labels.user_id.eq(user)].merge(names,on="product_id")
        actual["是否历史买过"] = actual.product_id.isin(history.product_id)
        actual["是否命中清单"] = actual.product_id.isin(top.product_id)
        st.dataframe(actual[["product_name","是否历史买过","是否命中清单"]],hide_index=True)
        result=data["per_user"].set_index("user_id").loc[user]
        st.write(f"复购目标 {int(result.repeat_targets)} 种，命中 {int(result.hits)} 种。")
        st.caption("第一次购买的商品不属于本版本候选范围；未购买不代表忘买。")
with tabs[3]:
    st.write("固定用户划分：训练 20,000，验证 5,000，最终确认 8,000。" if is_study else "固定用户划分：训练 70%，验证 15%，测试 15%。")
    st.write("所有模型在同一候选范围内评价。")
    st.dataframe(data["metrics"],hide_index=True)
    if is_study:
        st.image(str(ROOT/"reports/figures/information_value.png"))
        st.image(str(ROOT/"reports/figures/rules_coverage.png"))
    else:
        st.image(str(run/"figures/test_recall.png"))
        st.image(str(run/"figures/history_subgroups.png"))
    st.write("Recall@10 与 NDCG@10 对有复购目标的用户求平均；无复购目标订单单独计数。候选少于 10 时返回全部候选。")
    st.write("测试用户未参与训练，但其已有历史可作为输入。数据没有绝对日期，本实验不属于全平台日历时间回测。")
