"""Local, read-only demonstration using frozen held-out predictions."""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Next Basket Lab", page_icon="🛒", layout="wide")
st.title("下一单复购清单预测")
st.caption("从历史购买记录生成候选清单，比较个人习惯、用户分群与商品关联。")
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


data = load_tables(str(cache),str(run),is_study,selected["name"])
tabs = st.tabs(["最新补充研究", "购物清单演示", "第二轮实验结果", "第二轮研究发现"])
with tabs[0]:
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
        st.write("六套分组设置都检查了三种比较：全体用户、规模重叠的同一批用户、这批用户统一权重后。排除用户与重新加权都改变结果；这些是描述性诊断，不是历史长度的因果效应。")
        st.warning("需要修正第二轮解释：规则覆盖内的局部收益未在新用户中稳定复现。")
        st.image(str(extension/"figures/quota_diagnostic.png"))
        st.write("为减少规则挤占推荐名额，尝试保留原模型的覆盖名额再排序。验证集选择不启用该改动，因此不能宣称新方法提高了预测表现。")
        st.download_button("下载补充研究证据稿",(extension/"research_extension.md").read_text(encoding='utf-8'),file_name='research_extension.md',mime='text/markdown')
    else:
        st.info("补充研究证据稿尚未生成；已完成的第二轮研究在其他标签页。")
with tabs[3]:
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
with tabs[1]:
    st.caption("本页保留第二轮验证选定模型的购物清单案例；最新基线与稳健性结果见第一个标签页。")
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
with tabs[2]:
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
