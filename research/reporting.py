from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment import load_json


LABELS={"frequency":"All-history frequency","decay":"Recency-weighted frequency","window3":"Last 3 orders",
        "window5":"Last 5 orders","window10":"Last 10 orders","frequency_context":"Frequency + context",
        "frequency_recency":"+ recency","no_temporal":"Without temporal features","no_cadence":"Without cadence",
        "no_global":"Without population statistics","full":"Full personal-history model","co_narrow":"+ narrow co-purchase rules",
        "co_wide":"+ broad co-purchase rules","sequential":"+ next-order rules","cluster3":"+ K=3 clusters",
        "cluster6":"+ K=6 clusters","cluster10":"+ K=10 clusters"}


def md_table(frame,precision=4):
    def cell(x):
        if isinstance(x,(float,np.floating)):return f"{x:.{precision}f}" if np.isfinite(x) else "—"
        return str(x)
    return "| "+" | ".join(frame.columns)+" |\n|"+"|".join(["---"]*len(frame.columns))+"|\n"+"\n".join("| "+" | ".join(map(cell,row))+" |" for row in frame.itertuples(index=False,name=None))


def create_report(config,cache,run):
    if not (run/"audit.json").exists():raise FileNotFoundError("Independent audit must pass first")
    audit=load_json(run/"audit.json")
    if audit["status"]!="PASS":raise ValueError("Audit failed")
    out=Path("reports");figdir=out/"figures";tables=out/"tables"
    figdir.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    dest=run/"confirmation";m=pd.read_csv(dest/"metrics.csv").set_index("name")
    v=pd.read_csv(run/"validation/metrics.csv").set_index("name")
    p=pd.read_csv(dest/"primary_contrasts.csv");r=pd.read_csv(dest/"rule_diagnostics.csv")
    h=pd.read_csv(dest/"history_raw.csv");s=pd.read_csv(dest/"history_standardized.csv")
    w=pd.read_csv(dest/"history_window_comparisons.csv");scope=load_json(dest/"scope.json")
    support=load_json(dest/"history_common_support.json");sel=load_json(run/"selection.json")
    cluster=pd.read_csv(cache/"cluster_stability.csv")
    decomposition=pd.read_csv(run/"exploratory/rule_hit_decomposition.csv")
    shutil.copyfile(run/"exploratory/rule_hit_decomposition.csv",tables/"rule_hit_decomposition.csv")
    shutil.copyfile(run/"exploratory/provenance.json",out/"exploratory_provenance.json")
    for path in dest.glob("*.csv"):shutil.copyfile(path,tables/path.name)
    shutil.copyfile(cache/"cluster_stability.csv",tables/"cluster_stability.csv")
    shutil.copyfile(run/"validation/metrics.csv",tables/"validation_metrics.csv")
    for path in [run/"audit.json",dest/"scope.json",dest/"history_common_support.json",run/"selection.json"]:
        shutil.copyfile(path,out/path.name)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180})
    def save(fig,name):
        fig.savefig(figdir/f"{name}.png",bbox_inches="tight")
        fig.savefig(figdir/f"{name}.svg",bbox_inches="tight")
        plt.close(fig)
    names=["frequency","decay","frequency_context","frequency_recency","no_temporal","no_cadence","no_global","full"]
    fig,ax=plt.subplots(figsize=(10,5.5));y=np.arange(len(names))
    ax.barh(y,m.loc[names,"recall_at_10"]*100,color=["#94a3b8"]*7+["#0f766e"])
    ax.set_yticks(y,[LABELS[n] for n in names]);ax.invert_yaxis();ax.set_xlim(0,70)
    for j,n in enumerate(names):ax.text(m.loc[n,"recall_at_10"]*100+.5,j,f"{m.loc[n,'recall_at_10']:.2%}",va="center")
    ax.set_xlabel("Repeat-target Recall@10 (%)");ax.set_title("Information value on 8,000 fresh confirmation users")
    fig.tight_layout();save(fig,"information_value")
    fig,ax=plt.subplots(figsize=(11,5.8));y=np.arange(len(p));delta=p.delta*100
    ax.errorbar(delta,y,xerr=np.vstack([delta-p.sim_low*100,p.sim_high*100-delta]),fmt="o",color="#0f766e",capsize=4)
    ax.set_yticks(y,[f"{a} minus {b}" for a,b in zip(p.left,p.right)]);ax.invert_yaxis()
    ax.axvline(0,color="gray",lw=1);ax.axvline(.5,color="#d97706",ls="--",label="Predeclared practical gain: +0.5 pp")
    ax.set_xlabel("Recall@10 difference (percentage points)");ax.set_title("Eight primary contrasts: simultaneous 95% bootstrap intervals")
    ax.legend(loc="lower right",fontsize=9);fig.tight_layout();save(fig,"primary_contrasts")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));x=np.arange(3)
    axes[0].bar(x,h.recall*100,color="#64748b");axes[0].set_xticks(x,h.history_group)
    axes[0].set_title("Unadjusted, nonempty repeat targets")
    axes[1].bar(x,s.standardized_recall*100,color="#0f766e");axes[1].set_xticks(x,s.history_group)
    axes[1].set_title(f"Same candidate/target-size cell weights\nCommon support: {support['included_users']:,} users")
    for ax in axes:ax.set_ylim(0,80);ax.set_xlabel("Number of historical orders");ax.set_ylabel("Recall@10 (%)")
    fig.suptitle("Descriptive standardization; this is not a causal history-length effect",fontsize=12)
    fig.tight_layout();save(fig,"history_difficulty")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));x=np.arange(len(r))
    axes[0].bar(x-.18,r.candidate_coverage*100,.36,label="Candidates");axes[0].bar(x+.18,r.target_coverage_micro*100,.36,label="Repeat targets")
    axes[0].set_xticks(x,r.rules,rotation=15);axes[0].set_ylabel("Coverage (%)");axes[0].legend();axes[0].set_title("Broader mining increases coverage")
    axes[1].errorbar(x,r.delta*100,yerr=np.vstack([(r.delta-r.ci95_low)*100,(r.ci95_high-r.delta)*100]),fmt="o",capsize=4,color="#0f766e")
    axes[1].axhline(0,color="gray");axes[1].set_xticks(x,r.rules,rotation=15);axes[1].set_ylabel("Covered-target Recall@10 difference (pp)")
    axes[1].set_title("Exploratory: rules versus full model\nMarginal 95% intervals; original rankings")
    fig.tight_layout();save(fig,"rules_coverage")
    display=m[["recall_at_10","ndcg_at_10","precision_at_10","f1_at_10","seed_recall_min","seed_recall_max"]].copy()
    display.insert(0,"validation_recall",v.recall_at_10)
    display=display.reset_index()
    primary=p[["left","right","delta","ci95_low","ci95_high","sim_low","sim_high"]].copy()
    for col in primary.columns[2:]:primary[col]*=100
    primary=primary.rename(columns={x:x+"_pp" for x in primary.columns[2:]})
    cadence=p[(p.left=="full")&(p.right=="no_cadence")].iloc[0]
    temporal=p[(p.left=="full")&(p.right=="no_temporal")].iloc[0]
    global_row=p[(p.left=="full")&(p.right=="no_global")].iloc[0]
    rule_rows=p[p.left.isin(["co_wide","sequential",sel["selected_cluster"]])]
    raw_gap=float(h.iloc[0].recall-h.iloc[-1].recall)
    adj_gap=float(s.iloc[0].standardized_recall-s.iloc[-1].standardized_recall)
    def evidence(row):
        if row.sim_low>.005:return "支持超过预设 0.5 个百分点的正向收益"
        if row.sim_low>0:return "支持正向收益，但尚不能确认超过预设 0.5 个百分点"
        if row.sim_high<.005:return "未确认正向收益，且区间上界低于预设 0.5 个百分点"
        return "方向或实际大小仍不确定"
    text=f'''# Instacart 复购预测：哪些信息有用，复杂特征何时没有增益？

本文件是研究证据稿，供小组复核、讨论和自行撰写课程报告使用。全部数值来自本机真实运行，不是预设结果。

## 主要发现

1. **近期行为比累计频次更有预测价值。** 频次基线 Recall@10 为 {m.loc['frequency','recall_at_10']:.2%}，指数衰减频次为 {m.loc['decay','recall_at_10']:.2%}，完整个人历史模型为 {m.loc['full','recall_at_10']:.2%}。移除全部时间相关特征后，完整模型的优势为 {temporal.delta*100:.2f} 个百分点；{evidence(temporal)}。
2. **贡献可以拆开检验。** 购买节奏特征的条件增益为 {cadence.delta*100:.2f} 个百分点，{evidence(cadence)}；全局商品统计的条件增益为 {global_row.delta*100:.2f} 个百分点，{evidence(global_row)}。这是固定模型与其他特征后的消融，不能解释为因果影响或各组信息的绝对价值。
3. **本次规则和聚类没有有实际意义的总体增益，但规则存在局部信号。** 更宽共购规则、跨订单规则、验证集选定的聚类，三组相对完整模型的差值分别为 {', '.join(f'{x:+.2f}' for x in rule_rows.delta*100)} 个百分点；调整区间的上界均小于预设 +0.5 个百分点。窄共购规则在覆盖目标上的 Recall@10 提升约 {r.iloc[0].delta*100:.2f} 个百分点，但总体没有收益。不能据此说所有规则无用，也不能把局部提升当成整体进步。
4. **“历史越长，模型越差”的直接解释不成立。** 原始短历史组 {h.iloc[0].recall:.2%}、长历史组 {h.iloc[-1].recall:.2%}；限制共同支持并统一候选数、复购目标数的区间权重后，短组 {s.iloc[0].standardized_recall:.2%}、长组 {s.iloc[-1].standardized_recall:.2%}，排序反转。对比改变了目标总体，分箱也较粗；这揭示了总体比较的混杂，不能证明增加历史会提高多少表现。

![信息增益](figures/information_value.png)

## 设计与问题边界

新研究排除第一轮全部 10,000 名用户，再固定抽取 33,000 名：20,000 训练、5,000 验证、8,000 最终确认。第一轮用于提出问题，不再作为独立验证。新划分种子为 20261109；每个学习模型用 42、73、123 三个训练种子，所有消融共享参数；确定性基线只运行一次。

主要目标是**下一单中历史买过商品的排序**。候选包含每位用户全部历史商品，无负采样。每用户最后一单作目标，其前全部订单作为历史；训练、验证、确认用户互不重叠。全局统计、分群、规则仅拟合训练用户历史，确认用户自己的过去记录可作为输入。下一单日期与间隔不作为特征。该数据没有绝对日期，因此并非全平台日历回测。

确认组有 {scope['users']:,} 名用户，其中 {scope['empty_repeat_targets']:,} 人没有复购目标。主指标对非空复购目标用户求平均；Precision/F1 包含所有用户。表中学习模型是三个种子的指标均值，**不是分数集成模型**。同分用商品 ID 排序，候选不足 10 时输出全部。

候选范围对整张下一单的平均覆盖上限为 {scope['candidate_recall_ceiling_macro']:.2%}；完整模型的整篮 Recall@10 为 {scope['full_model_whole_basket_recall_at_10']:.2%}，与复购 Recall@10 的 {scope['full_model_repeat_recall_at_10']:.2%} 分母不同。不能把后者称为整个购物篮的“准确率”。

## 实验结果

验证集选定 `{sel['selected']}`，聚类备选选定 `{sel['selected_cluster']}`，窗口基线选定 `{sel['selected_window']}`。选择在读取确认标签前固定；确认集没有用于重选模型。三种种子只反映这些训练随机性，未重复抽样训练用户。

{md_table(display)}

## 主要对比及不确定性

事先声明 8 组主要对比、实际意义阈值 +0.5 个百分点。按用户配对 bootstrap 6,000 次，在用户层先平均三种种子的指标。下表同时给出边际 95% 区间与对 8 个比较进行 Bonferroni 调整的近似同时 95% 区间。区间条件于本次训练样本和三个已拟合模型，不包含重新抽样训练用户的不确定性。所有差值均以百分点表示。

{md_table(primary,3)}

![主要对比](figures/primary_contrasts.png)

## 为什么关联规则未必有用？

窄规则在最常见 500 商品上挖掘；宽规则和跨订单规则扩大至 2,000 商品，同时降低支持度门槛并限制每个前件最多 50 条规则，因此这是**整套覆盖方案**的比较，不能把变化单独归因于商品数。跨订单规则只统计同一用户相邻订单，排除商品指向自身。长度为二的项集使用精确稀疏共现计数，已用人工小数据验证与 FP-Growth 一致。

规则由用户最后一张已观察订单触发。覆盖到目标后，仍从原完整候选排序中计算命中；不缩小候选池制造更高分。以下条件分析为探索性，使用边际区间；不能用它取代总体主要检验。

{md_table(r[['rules','candidate_coverage','target_coverage_micro','users_any_covered_target','full_covered_recall','rule_covered_recall','delta','ci95_low','ci95_high']])}

![规则覆盖与效果](figures/rules_coverage.png)

共购规则覆盖内确有局部信号：窄规则的条件差值区间不含零；宽规则的局部均值也为正，幅度较小；跨订单规则区间包含零。它们与总体无收益可以同时成立，因为覆盖内指标与总体指标的分母、参与用户不同。

为解释这个现象，在确认完成之后增加了**事后探索性分解**，没有再训练或重选模型。对同一批非空目标用户，把总 Recall 差值分为“规则覆盖目标的命中变化 / 全部复购目标数”和“未覆盖目标的命中变化 / 全部复购目标数”。两项恰好相加为总体变化，不能与上表覆盖内 Recall 差值直接相加。

{md_table(decomposition[['rules','covered_recommendations_per_user_delta','covered_delta','uncovered_delta','total_delta']],6)}

窄共购规则将更多推荐名额给覆盖商品：覆盖部分贡献 +{decomposition.iloc[0].covered_delta*100:.3f} 个百分点，未覆盖部分贡献 {decomposition.iloc[0].uncovered_delta*100:.3f} 个百分点，净值 {decomposition.iloc[0].total_delta*100:.3f} 个百分点。这个算术分解表明，本次局部收益没有抵消其他目标命中的损失。宽规则的两部分区间仍含零，不应将其变化写成确定机制；全部区间见 `tables/rule_hit_decomposition.csv`。这是对冻结排序的描述，不是用户行为的因果实验，也不是提前指定的主要检验。

已有个人频次与近期特征可能吸收了部分规则信息，但本实验未直接测量信息冗余，不能把这个解释写成已证实机制。规则可作为展示关系的描述工具，不能冒充模型采用该建议的解释。

聚类轮廓系数范围 {cluster.silhouette.min():.3f}–{cluster.silhouette.max():.3f}，说明这些标准化画像下群体分离较弱；种子间 ARI 见 `tables/cluster_stability.csv`。较高 ARI 表示算法可重复，不能证明存在清晰的真实人群类别，也不能证明弱分离造成预测无增益。

## 历史长度与任务难度

{md_table(h)}

标准化使用候选数 × 复购目标数二维分箱，要求每个历史组每格至少 10 人，按共同支持总体格子频率赋相同权重。共有 {support['cells']} 格、{support['included_users']:,}/{support['nonempty_users']:,} 名非空目标用户保留。其余用户被排除，因此标准化结果只适用于重叠人群。真实目标数只用于事后诊断，不进入推荐模型。

{md_table(s)}

![历史与难度](figures/history_difficulty.png)

额外在同一批历史至少 16 单用户上固定候选池与下一单目标，仅改变频次计数使用的最近订单窗口，避免不同用户构成干扰。以下为探索性比较，不能把截断计数窗口解释为用户只拥有那么多历史，也不能由此声称“删掉旧数据必然更好”。

{md_table(w[['model','users','delta','ci95_low','ci95_high']])}

K=5/20 的敏感性结果保存在 `tables/k_sensitivity.csv`。它们是预先声明的辅助指标，不用于重选模型。

同用户窗口实验提示：最近三单计数没有优于累计频次，而衰减加权有稳定优势；给旧信息逐渐降低权重，比笼统地“只留最新几单”更有依据。这里的周期消融只涉及已实现的订单间隔、间隔波动与间隔比例，不代表真实补货周期不重要；下一次购物的时间本身在当前预测时点未知。

## 对项目与产品的含义

项目的中心贡献是**增量信息与评估边界的实证分析**，不以刷新排行榜或发明新算法为贡献。建议标题为“个人近期行为与群体关联：下一单复购预测中的信息增益及边界”。已有研究已经强调复购与探索的区分，本项目应定位为课程级复现、消融与边界检查，而非首次发现该现象。[Li et al., A Next Basket Recommendation Reality Check](https://arxiv.org/abs/2109.14233)

实现可用于“可能再次购买”的短清单。它不能判断用户忘买了什么，不能替代新品推荐，没有实验表明能提高销售或节省购物时间。若将来做整篮推荐，需要增加新商品候选生成与独立评价，而非仅提高复购排序分数。

## 复核、复现与局限

独立审核 `{audit['status']}`：核对 {audit['frozen_files']} 个冻结文件与 6 个原始文件；直接从原始 CSV 重建确认组的 {audit['candidate_rows_checked']:,} 个候选和 {audit['raw_target_rows_checked']:,} 个目标记录；以独立集合运算核对排序和 Recall、Precision、F1、NDCG。原始数据未修改。详见 `audit.json`。

完整步骤见根目录 README。`research/protocol.json` 是确认前方案，`autoresearch/research-v2/freeze.json` 保存模型、选择和输入哈希，逐种子模型与分数保存在运行目录中。主要结果表与图在本报告目录下。

局限包括：单个历史平台与有标签用户子样本；单一训练用户划分；只预测复购；固定长度清单；商品价格和绝对时间缺失；规则参数与表示覆盖范围有限；没有在线用户实验。多个种子不能替代跨数据集或跨时期复现。统计上很小的收益不自动等于产品收益。

课程要求的完整流程、比较、消融、讨论与演示均有可核查材料。课程 PDF 是示范与要求来源，不能从示范推断必须堆叠每一种算法。小组仍需根据正式 rubric 核对格式、页数、署名和引用，自己理解并完成最终报告、海报与答辩。

研究方案、实现、图表及本证据稿由 Codex 辅助生成，并实际运行验证；应如实披露 AIGC 用途，不将本稿直接作为独立完成的最终提交材料。数据下载来源为 [Kaggle 镜像](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)，文件指纹保存在缓存清单中；仅凭 CSV 不能识别镜像页面的发布修订号。
'''
    (out/"research_findings.md").write_text(text,encoding="utf-8")
    return {"report":str(out/"research_findings.md"),"figures":4,"audit":audit["status"]}
