"""Local, read-only demonstration using frozen held-out predictions."""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Next Basket Lab", page_icon="🛒", layout="wide")
st.title("下一单复购清单预测")
st.caption("从历史购买记录生成候选清单，比较个人习惯、用户分群与商品关联。")
pointer = ROOT / "outputs/latest_run.txt"
if not pointer.exists():
    st.info("请先运行数据准备、实验循环和最终测试。操作说明见项目 README。")
    st.stop()
run = Path(pointer.read_text(encoding="utf-8").strip())
if not (run/"test/completed.json").exists():
    st.info("实验尚未完成最终测试。完成后这里会自动展示结果。")
    st.stop()
config = json.loads((run/"config.json").read_text(encoding="utf-8"))
cache = ROOT / config["cache"]
selected = json.loads((run/"selected.json").read_text(encoding="utf-8"))


@st.cache_data
def load_tables(cache_path, run_path):
    c, r = Path(cache_path), Path(run_path)
    return {
        "users": pd.read_parquet(c/"users.parquet"),
        "history": pd.read_parquet(c/"history.parquet"),
        "products": pd.read_parquet(c/"products.parquet"),
        "clusters": pd.read_parquet(c/"clusters.parquet"),
        "rules": pd.read_csv(c/"rules.csv"),
        "predictions": pd.read_parquet(r/"test/selected_predictions.parquet"),
        "metrics": pd.read_csv(r/"test/metrics.csv"),
        "per_user": pd.read_csv(r/f"test/{selected['name']}_users.csv"),
    }


data = load_tables(str(cache),str(run))
tabs = st.tabs(["购物清单演示", "实验结果与边界"])
with tabs[0]:
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
        st.caption(f"验证集选定模型：{selected['name']}。分数用于排序，不宣称已校准为概率。")
        rules = data["rules"]
        last = history.loc[history.age.eq(0),"product_id"]
        matched = rules[rules.antecedent.isin(last) & rules.consequent.isin(top.product_id)].nlargest(5,"confidence")
        if len(matched):
            st.write("相关历史共购规则（不代表因果，也不一定是模型采用的理由）：")
            mapping=names.set_index("product_id").product_name
            for row in matched.itertuples():
                st.write(f"{mapping.get(row.antecedent)} → {mapping.get(row.consequent)}；置信度 {row.confidence:.1%}，提升度 {row.lift:.2f}")
    if st.button("揭晓真实下一单"):
        labels = pd.read_parquet(cache/"labels_test.parquet")
        actual = labels[labels.user_id.eq(user)].merge(names,on="product_id")
        actual["是否历史买过"] = actual.product_id.isin(history.product_id)
        actual["是否命中清单"] = actual.product_id.isin(top.product_id)
        st.dataframe(actual[["product_name","是否历史买过","是否命中清单"]],hide_index=True)
        result=data["per_user"].set_index("user_id").loc[user]
        st.write(f"复购目标 {int(result.repeat_targets)} 种，命中 {int(result.hits)} 种。")
        st.caption("第一次购买的商品不属于本版本候选范围；未购买不代表忘买。")
with tabs[1]:
    st.write("固定用户划分：训练 70%，验证 15%，测试 15%。所有模型在同一候选范围内评价。")
    st.dataframe(data["metrics"],hide_index=True)
    st.image(str(run/"figures/test_recall.png"))
    st.image(str(run/"figures/history_subgroups.png"))
    st.write("Recall@10 与 NDCG@10 对有复购目标的用户求平均；无复购目标订单单独计数。候选少于 10 时返回全部候选。")
    st.write("测试用户未参与训练，但其已有历史可作为输入。数据没有绝对日期，本实验不属于全平台日历时间回测。")
