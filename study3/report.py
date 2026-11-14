"""Evidence report for the completed extension; no fitting or model selection."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAMES = {
    "frequency": "Personal frequency", "decay": "Decayed frequency",
    "full": "Full history model", "co_narrow": "+ co-purchase rules",
    "no_cadence": "Without cadence", "quota": "Selected quota policy",
    "tifu": "TIFU-KNN adaptation", "tifu_self": "TIFU personal-only control",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(frame, precision=4):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{precision}f}" if np.isfinite(value) else "—"
        return str(value)
    header = "| "+" | ".join(map(str, frame.columns))+" |\n"
    header += "|"+"|".join(["---"]*len(frame.columns))+"|\n"
    return header+"\n".join("| "+" | ".join(cell(v) for v in row)+" |" for row in frame.itertuples(index=False, name=None))


def verdict(row, threshold):
    if row.sim_low > threshold:
        return "调整后的区间支持超过预设实际意义阈值的正向收益"
    if row.sim_low > 0:
        return "调整后的区间支持正向收益，但尚不能确认超过预设实际意义阈值"
    if row.sim_high < 0:
        return "调整后的区间支持负向差异，结果不利于左侧方法"
    if row.sim_high < threshold:
        return "没有确认正向收益，且调整后的区间上界低于预设实际意义阈值"
    return "区间仍包含零及有实际意义的正向收益，当前证据未能区分"


def report(cache, run):
    cache, run = Path(cache), Path(run)
    if not (run/"confirmation/completed.json").exists():
        raise RuntimeError("Do not inspect confirmation evidence before completion")
    audit = read(run/"audit.json")
    if audit.get("status") != "PASS":
        raise RuntimeError("Independent audit must pass before reporting")
    frozen = read(run/"freeze.json"); protocol = frozen["protocol"]
    selection = read(run/"selection.json"); dest = run/"confirmation"
    metrics = pd.read_csv(dest/"metrics.csv").set_index("name")
    valid = pd.read_csv(run/"validation/metrics.csv").set_index("name")
    contrasts = pd.read_csv(dest/"primary_contrasts.csv")
    decomposition = pd.read_csv(dest/"rule_decomposition.csv")
    new_covered = decomposition[decomposition.name.eq("co_narrow")].iloc[0]
    quota_grid = pd.read_csv(run/"quota_validation.csv")
    tifu_grid = pd.read_csv(run/"tifu_validation.csv")
    scope = read(dest/"scope.json")
    threshold = protocol["practical_threshold"]
    weight = selection["quota_weight"]
    chosen = selection["tifu"]
    history = {name: {"summary": read(dest/f"history_{name}/summary.json"),
                      "sensitivity": pd.read_csv(dest/f"history_{name}/sensitivity.csv"),
                      "groups": pd.read_csv(dest/f"history_{name}/groups.csv")}
               for name in protocol["history_models"]}
    out = Path("reports/study3"); figs = out/"figures"; tables = out/"tables"
    figs.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    for path in dest.glob("*.csv"):
        shutil.copyfile(path, tables/path.name)
    for name in ["quota_validation.csv", "tifu_validation.csv"]:
        shutil.copyfile(run/name, tables/name)
    old_rule_path = Path("autoresearch/research-v2/confirmation/rule_diagnostics.csv")
    old_rule = None
    if old_rule_path.exists():
        old_rule = pd.read_csv(old_rule_path).set_index("rules").loc["co_narrow"]
        shutil.copyfile(old_rule_path, tables/"rule_diagnostics_v2_exploratory.csv")
    shutil.copyfile(run/"validation/metrics.csv", tables/"validation_metrics.csv")
    for name in protocol["history_models"]:
        target = tables/f"history_{name}"; target.mkdir(exist_ok=True)
        for pattern in ["*.csv", "*.json", "method.txt"]:
            for path in (dest/f"history_{name}").glob(pattern):
                shutil.copyfile(path, target/path.name)
    for path in [run/"audit.json", run/"selection.json", dest/"scope.json"]:
        shutil.copyfile(path, out/path.name)
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    for name in ["tifu_sources.md", "quota_method.md", "history_notes.txt"]:
        shutil.copyfile(Path("study3")/name, out/name)
    previous = Path("outputs/study3_history_v2_exploratory")
    old_history = None
    if (previous/"summary.json").exists():
        target = tables/"history_v2_full_exploratory"; target.mkdir(exist_ok=True)
        for filename in ["summary.json", "sensitivity.csv", "groups.csv"]:
            shutil.copyfile(previous/filename, target/filename)
        old_history = read(previous/"summary.json")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    def save(fig, name):
        fig.savefig(figs/f"{name}.png", bbox_inches="tight")
        svg = figs/f"{name}.svg"
        fig.savefig(svg, bbox_inches="tight")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
        plt.close(fig)

    names = ["frequency", "decay", "tifu_self", "tifu", "full", "no_cadence", "co_narrow", "quota"]
    fig, ax = plt.subplots(figsize=(10, 5.2)); y = np.arange(len(names))
    ax.barh(y, metrics.loc[names, "recall_at_10"]*100,
            color=["#94a3b8", "#94a3b8", "#0e7490", "#0e7490", "#0f766e", "#64748b", "#d97706", "#d97706"])
    labels = [NAMES[name]+(" (fallback to base)" if name == "quota" and weight == 0 else "") for name in names]
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xlim(0, min(100, max(65, metrics.recall_at_10.max()*100+10)))
    for j, name in enumerate(names):
        ax.text(metrics.loc[name, "recall_at_10"]*100+.6, j, f"{metrics.loc[name, 'recall_at_10']:.2%}", va="center")
    ax.set_xlabel("Repeat-target Recall@10 (%)")
    ax.set_title(f"Fresh confirmation: {scope['users']:,} users / {scope['nonempty_users']:,} nonempty targets")
    fig.tight_layout(); save(fig, "domain_baseline")

    fig, ax = plt.subplots(figsize=(11, 5.5)); y = np.arange(len(contrasts))
    delta = contrasts.delta.to_numpy()*100
    ax.errorbar(delta, y, xerr=np.vstack([delta-contrasts.sim_low*100, contrasts.sim_high*100-delta]),
                fmt="o", color="#0f766e", capsize=4)
    labels = [f"{a} minus {b}" for a, b in zip(contrasts.left, contrasts.right)]
    if weight == 0:
        labels = [label+(" [identity control]" if label == "quota minus full" else "") for label in labels]
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.axvline(0, color="#6b7280", lw=1)
    ax.axvline(threshold*100, color="#d97706", ls="--", label=f"Practical gain threshold: +{threshold*100:g} pp")
    ax.set_xlabel("Recall@10 difference (percentage points)")
    ax.set_title(f"{len(contrasts)} predeclared contrasts: simultaneous 95% bootstrap intervals")
    ax.legend(loc="best", fontsize=9); fig.tight_layout(); save(fig, "primary_contrasts")

    fig, axes = plt.subplots(len(history), 1, figsize=(11, 3.1*len(history)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, record) in zip(axes, history.items()):
        rows = record["sensitivity"]; x = np.arange(len(rows)); scale = 100
        ax.axhline(0, color="gray", lw=1)
        ax.axhline(rows.raw_overall_long_minus_short.iloc[0]*scale, color="#94a3b8", ls="--", label="Raw, all nonempty users")
        ax.plot(x-.07, rows.raw_common_support_long_minus_short*scale, "s", color="#d97706", label="Raw, same common-support users")
        mean = rows.standardized_long_minus_short*scale
        ax.errorbar(x+.07, mean, yerr=np.vstack([mean-rows.standardized_ci95_low*scale, rows.standardized_ci95_high*scale-mean]),
                    fmt="o", color="#0f766e", capsize=3, label="Same users, standardized weights (95% CI)")
        ax.set_ylabel("Long minus short (pp)"); ax.set_title(NAMES[name])
        if ax is axes[0]:
            ax.legend(loc="best", fontsize=8, ncol=1)
        ax.set_xticks(x, [s.replace("_min", "\nmin=") for s in rows.specification])
    axes[-1].set_xlabel("Six prespecified support / binning settings")
    fig.suptitle("History-length robustness: descriptive only; intervals condition on support and weights", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .97)); save(fig, "history_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    grid_delta = (quota_grid.recall-valid.loc["full", "recall_at_10"])*100
    axes[0].plot(quota_grid.weight, grid_delta, "o-", color="#0e7490")
    chosen_row = quota_grid[quota_grid.weight.eq(weight)].iloc[0]
    axes[0].scatter([weight], [(chosen_row.recall-valid.loc["full", "recall_at_10"])*100], s=85, facecolors="none", edgecolors="#d97706", linewidths=2)
    axes[0].axhline(0, color="gray", lw=1); axes[0].set_xticks(quota_grid.weight)
    axes[0].set_xlabel("Rule score mixing weight"); axes[0].set_ylabel("Validation Recall@10 change (pp)")
    axes[0].set_title(f"Validation chooses weight={weight:g}")
    x = np.arange(len(decomposition)); width = .25
    for offset, col, color, label in [(-width/2, "covered_contribution", "#0f766e", "Covered targets"),
                                       (width/2, "uncovered_contribution", "#d97706", "Uncovered targets")]:
        values = decomposition[col+"_delta"]*100
        errors = np.vstack([(decomposition[col+"_delta"]-decomposition[col+"_ci95_low"])*100,
                            (decomposition[col+"_ci95_high"]-decomposition[col+"_delta"])*100])
        axes[1].bar(x+offset, values, width, yerr=errors, color=color, label=label, capsize=3)
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].set_xticks(x, ["Rule model", "Selected quota\n"+("base identity" if weight == 0 else f"weight={weight:g}")])
    axes[1].set_ylabel("Confirmation contribution change (pp)")
    axes[1].set_title("Shared total-target denominator\nExploratory marginal 95% intervals"); axes[1].legend(fontsize=8)
    fig.tight_layout(); save(fig, "quota_diagnostic")

    def contrast(a, b):
        return contrasts[(contrasts.left == a) & (contrasts.right == b)].iloc[0]
    tifu_gain = contrast("tifu", "tifu_self"); full_tifu = contrast("full", "tifu")
    rule_gain = contrast("co_narrow", "full"); quota_gain = contrast("quota", "full")
    quota_rule = contrast("quota", "co_narrow"); cadence = contrast("no_cadence", "full")
    primary = contrasts[["left", "right", "delta", "ci95_low", "ci95_high", "sim_low", "sim_high"]].copy()
    for column in primary.columns[2:]:
        primary[column] *= 100
    primary = primary.rename(columns={column: column+"_pp" for column in primary.columns[2:]})
    display = metrics[["recall_at_10", "ndcg_at_10", "precision_at_10", "f1_at_10"]].copy()
    display.insert(0, "validation_recall", valid.recall_at_10)
    display = display.reset_index()
    history_rows = []
    for name, value in history.items():
        summary = value["summary"]
        history_rows.append({"方法": name, "有效规格": summary["evaluable_specifications"],
                             "标准化差值最小_pp": summary["standardized_delta_min"]*100 if summary["standardized_delta_min"] is not None else np.nan,
                             "标准化差值最大_pp": summary["standardized_delta_max"]*100 if summary["standardized_delta_max"] is not None else np.nan,
                             "六规格CI下界均正": summary["all_six_standardized_ci_lower_positive"],
                             "六规格同支持内均反转": summary["all_six_reversed_within_common_support"],
                             "保留用户最少": summary["retained_user_min"], "保留用户最多": summary["retained_user_max"]})
    history_display = pd.DataFrame(history_rows)
    history_display.to_csv(tables/"history_summary.csv", index=False)
    history_statements = []
    for name, value in history.items():
        summary = value["summary"]
        positive = "六套规格的区间下界均为正" if summary["all_six_standardized_ci_lower_positive"] else "没有出现六套规格的区间下界均为正的结果"
        same_support = "六套规格都在相同支持人群内发生方向反转" if summary["all_six_reversed_within_common_support"] else "并非六套规格都在相同支持人群内反转，不能把全部变化归因于重新加权"
        history_statements.append(f"- `{name}`：{positive}；{same_support}。")
    if weight == 0:
        quota_interpretation = (
            "验证集选中了 **w=0**，即保持原基础模型排序。四个候选权重里，非零权重没有取得更高的验证集平均 Recall。"
            "因此本轮没有采用非零规则重排；最终确认集 quota 与 full 的零差值是**定义上的恒等结果**，"
            "不能把它解释为独立证据，证明所有非零配额约束都没有价值。"
            "quota 相对普通规则模型的差值，此时就是基础模型相对普通规则模型的差值，不能归功于配额机制。"
            "本次有证据支持的结论是：所测试的约束与混合权重未在开发阶段赢得模型变更；停止调参并保留这个结果。"
        )
    else:
        quota_interpretation = (
            f"验证集选中 w={weight:g}；相对基础模型的确认差值为 {quota_gain.delta*100:+.3f} 个百分点，"
            f"{verdict(quota_gain, threshold)}。相对普通规则模型的差值为 {quota_rule.delta*100:+.3f} 个百分点。"
            "配额约束保持推荐名额的分组计数，但不保证召回率或其他目标不下降。"
        )
    if chosen["alpha"] == 1:
        neighbor_interpretation = "本次选择的个人权重为 1，实际没有使用邻居；tifu 与同设置的 tifu_self 相同，因此它们的零差值不是邻居无用的独立检验。"
    else:
        neighbor_interpretation = f"加入邻居相对同一时间设置的个人向量对照，差值为 {tifu_gain.delta*100:+.3f} 个百分点，{verdict(tifu_gain, threshold)}。这检验了当前邻居实现，而不是所有群体信息的价值。"
    if old_history is None:
        old_note = "旧 V2 的补充分箱输出未随本报告复制；其已发布分析属于提出本轮问题的探索阶段。"
    else:
        old_note = (
            f"旧 V2 的六规格事后分析中，标准化差值为 {old_history['standardized_delta_min']*100:.2f} 至 "
            f"{old_history['standardized_delta_max']*100:.2f} 个百分点；其输出另存 `tables/history_v2_full_exploratory/`。"
            "这是已经观察过的数据，不能算新的独立确认。旧研究更细分箱/min20 的同支持原始差值已经略为正，"
            "说明共同支持筛选本身也会改变人群比较，不能把所有方向变化只归因于加权。"
        )
    local_gain_note = (
        f"新确认数据中，窄规则覆盖目标的条件 Recall 差值为 {new_covered.covered_recall_delta*100:+.3f} 个百分点，"
        f"边际 95% 区间为 [{new_covered.covered_recall_ci95_low*100:+.3f}, {new_covered.covered_recall_ci95_high*100:+.3f}] 个百分点。"
    )
    if old_rule is not None:
        local_gain_note += (
            f"旧 V2 的对应探索性差值为 {old_rule.delta*100:+.3f} 个百分点，"
            f"边际区间为 [{old_rule.ci95_low*100:+.3f}, {old_rule.ci95_high*100:+.3f}]。"
        )
        if new_covered.covered_recall_ci95_low <= 0 <= new_covered.covered_recall_ci95_high:
            local_gain_note += (
                "**新样本没有复现旧样本中同样明确的局部正向证据**，所以当前不能继续把“关联规则局部有效”作为无条件主结论。"
                "这不等于证明两轮真实效应存在统计差异；它要求收窄旧发现的适用范围。"
            )
    all_history_positive = all(record["summary"]["all_six_standardized_ci_lower_positive"] for record in history.values())
    history_headline = (
        "三种方法的六套规格均得到正向标准化长组−短组差值，边际区间下界均为正，复现了对总体分组比较需要谨慎解释的发现。"
        if all_history_positive else
        "不同方法或规格的结果存在差异，需要逐一呈现，不能主张一个统一的历史长度规律。"
    )
    deriv = decomposition[["name", "covered_slots_delta", "covered_contribution_delta", "uncovered_contribution_delta", "total_delta"]].copy()
    for column in deriv.columns[2:]:
        deriv[column] *= 100
    deriv = deriv.rename(columns={column: column+"_pp" for column in deriv.columns[2:]})
    cadence_text = (
        f"移除本次实现的间隔与波动特征后，Recall 差值为 {cadence.delta*100:+.3f} 个百分点，"
        f"{verdict(cadence, threshold)}。比较方向是 no_cadence − full；不能把其符号误写成添加周期特征的收益。"
    )
    main = f'''# 研究扩展：领域基线、配额约束与用户分组评价的稳健性

本文件是课程项目的研究证据稿，供小组复核与独立撰写最终材料。记录真实运行结果，包括没有带来改进的设计；不作为可直接提交的最终课程报告。

## 本轮完成了什么

在先前两轮的 43,000 名用户之外，重新抽取 {sum(protocol[x] for x in ['train_users','validation_users','confirmation_users']):,} 名用户，分成 {protocol['train_users']:,} 训练、{protocol['validation_users']:,} 验证、{protocol['confirmation_users']:,} 最终确认。训练样本和确认样本均是新用户。方法和参数在打开确认标签前固定；原研究不被覆盖。

本轮有三个明确产出：补入 TIFU-KNN 领域代表基线及同设置个人向量对照；把“覆盖商品占用推荐名额”的解释转为可执行的配额约束；用六套固定分箱规格、三种不同方法复查历史长度比较。共报告 {len(metrics)} 种最终方法，学习模型的指标先在用户内平均 {len(protocol['model_seeds'])} 个训练种子；这不是分数集成。启发式与 TIFU-KNN 是确定性计算，单次运行。

1. **领域基线比较**：TIFU-KNN 适配版的确认 Recall@10 为 {metrics.loc['tifu','recall_at_10']:.2%}，完整历史模型为 {metrics.loc['full','recall_at_10']:.2%}。full − tifu 为 {full_tifu.delta*100:+.3f} 个百分点，{verdict(full_tifu, threshold)}。{neighbor_interpretation}因此先前的聚类/规则结果不能扩大解释为所有群体信息都没有用。
2. **提出并检验可解释的改进设计**：配额方案在验证集选择 w={weight:g}。{quota_interpretation}
3. **加强评价发现并收窄未复现的解释**：{history_headline}同时，关联规则覆盖内的改善必须重新检查，而不是沿用上一轮的解释。{local_gain_note}

![领域基线比较](figures/domain_baseline.png)

## 可主张的贡献及其边界

这是一项课程级实证扩展：从已观察的局部/总体差异提出一个可测试约束，再用预先规定的开发选择和新用户确认评估；同时交付一个可复用的评价诊断，区分人群筛选与重新加权造成的变化。**提出并检验设计不等于已经获得有效的新方法**。配额和重排的思想不宣称全球首次；未证明用户购买行为的因果机制，也未声称达到领域最佳结果。

单纯增加模型个数不是贡献。领域基线用于检查原结论是否仅由比较对象过弱造成；同支持诊断用于检查解释是否受人群构成影响；配额方案则使“推荐名额挤占”这一解释受到可以失败的检验。

## TIFU-KNN 的来源与公平适配

依据 Hu、He、Gao、Zhang 的 [SIGIR 2020 原论文](https://arxiv.org/abs/2006.00556) 与 [作者公开实现](https://github.com/HaojiHu/TIFUKNN/blob/master/TIFUKNN.py) 独立编写稀疏实现。方法把用户自己的时间衰减购买向量与欧氏距离最近邻的平均向量混合。只允许训练用户进入邻居库；验证、确认用户仅提供自己的历史，不能成为彼此的邻居。商品维度保留完整目录，最终排序仍限于每位用户历史买过的全部候选。

原论文描述和作者代码的分组/组内衰减存在差异。本轮显式采用 `author_code` 语义，并以独立密集循环与手算短历史检查等价性。具体差别、公式与参考来源见 [实现说明](tifu_sources.md)。这是**复购候选任务上的适配**，不能拿本地分数与论文整篮榜单直接比较。

验证集共检验 {len(tifu_grid)} 个组合：两种分组数、三种邻居数、四种个人权重；包含论文 Instacart 的 m=3、rb=0.9、rg=0.7、k=900、alpha=0.9 参数。选择结果是 m={selection['tifu_settings']['group_count']}、rb={selection['tifu_settings']['within_decay']}、rg={selection['tifu_settings']['group_decay']}、邻居数 {chosen['neighbors']}、个人权重 {chosen['alpha']}。tifu_self 使用**选定 TIFU 配置的同一时间向量**，用于隔离邻居的增量作用；它不是单独优化时间参数得到的最强个人基线。

{neighbor_interpretation}

**两个增量比较的起点不同**：邻居增益相对于 TIFU 的个人向量基线，规则增益相对于已经较强的完整 LightGBM 历史模型。因此不能把收益差异完全归因于信息表示，更不能据此证明邻居表示优于关联规则。较稳妥的表述是：群体信息的增量价值需要同时说明使用方式和已有个人模型基线。本轮没有运行“完整模型 + TIFU 邻居特征”的匹配对照，不能声称邻居信息仍能进一步改善完整模型。

全部验证组合见 `tables/tifu_validation.csv`，没有根据确认结果重新选择参数。

## 主要结果与七组预定比较

确认组有 {scope['users']:,} 人，其中 {scope['nonempty_users']:,} 人下一单有复购目标。Recall 与 NDCG 排除空复购目标用户；Precision 与 F1 包含所有人。候选少于 10 时返回全部候选；同分按商品 ID 排序。

{table(display)}

主要比较在用户层进行配对 bootstrap {protocol['bootstrap_replicates']:,} 次，并给出边际 95% 与对 {len(contrasts)} 个比较作 Bonferroni 调整的近似同时 95% 区间。实际意义阈值提前设为 +{threshold*100:g} 个百分点。区间条件于已经拟合的模型；不覆盖重新抽取训练用户的不确定性。以下差值均以百分点表示。

{table(primary, 3)}

![主要差值及区间](figures/primary_contrasts.png)

普通规则模型相对完整模型的差值为 {rule_gain.delta*100:+.3f} 个百分点，{verdict(rule_gain, threshold)}。{cadence_text}这些结果仅适用于实现过的特征与模型；不得据此宣布所有关联或真实补货周期无用。

## 配额约束：把上一轮解释变成可以失败的设计

对每位用户，先数出基础模型 top-K 中被规则覆盖的商品数 q，再保持 q 个覆盖名额、K−q 个未覆盖名额。各组内部按 `(1−w) × 基础分数 + w × 规则模型分数` 排序。覆盖标记和 q 都只来自历史与模型输出，不读取下一单标签。每个 K 都重新计算配额，候选不足 K 时选出全部候选。

该约束保证的是**覆盖/未覆盖的名额计数**，没有保证命中数。它仍可能在同组内换错商品，也可能阻止本来有益的跨组调整。w=0 是明确的“不改变基础结果”对照。完整方法与限制见 [配额设计](quota_method.md)。

如果未来进一步研究非零权重，还应加入“相同混合权重、没有配额限制”的匹配对照，才能把混合分数的作用与配额约束的作用分开。本轮未增加这个对照，也不据现有比较作机制归因。

{table(quota_grid)}

{quota_interpretation}

![配额验证与命中分解](figures/quota_diagnostic.png)

确认集的贡献分解沿用统一分母：每用户的全部复购目标数。覆盖部分和未覆盖部分相加恰好等于总 Recall 差值；这与“只在覆盖目标上的 Recall”不是同一个指标。此分析是探索性诊断，不替代主要比较。

{table(deriv, 4)}

本次规则候选覆盖比例为 {scope['covered_candidate_fraction']:.2%}，复购目标覆盖比例为 {scope['covered_target_fraction']:.2%}。完整边际区间、覆盖内 Recall 和每用户推荐名额变化均在 `tables/rule_decomposition.csv`。用户行为的因果机制不能由这些固定预测的算术分解推出。

{local_gain_note}

## 历史长度：可重复的诊断，不是因果调整

三个方法使用各自已经冻结的预测：完整历史模型、个人频次、TIFU-KNN。每个方法都运行 original/finer/coarser 分箱与每格每组至少 10/20 人的六套规格；不选择其中效果最好的规格。不同方法的共同支持人群一致，因为是否保留只由历史长度、候选数和真实复购目标数决定，不由模型表现决定。

{table(history_display, 3)}

{chr(10).join(history_statements)}

![六套历史长度诊断](figures/history_sensitivity.png)

图中灰虚线是所有非空目标用户的原始长组−短组差值；橙色方点是限制到共同支持后的同一批用户原始差值；绿色点是在这批相同用户上重新分配格子权重后的差值。**灰到橙是人群筛选变化，橙到绿才是同样本人群的权重变化**。二者不能混为“控制了规模后的因果效应”。每个方法的逐组人数、保留比例、所有格子均值和权重均随报告保存。

历史诊断区间使用 {history[protocol['history_models'][0]]['summary']['bootstrap_replicates']:,} 次格内用户 bootstrap，固定观察到的共同支持、格子人数和总体权重。因此区间反映这些条件下的用户抽样波动，不包含选择支持、估计权重或再训练的波动；六套规格是敏感性分析，不应当作六次独立发现。

真实目标大小是未来结果，仅用于事后评价，不能进入预测模型；对它作条件化可能引入选择效应，粗分箱也不能消除格内差异。若结果依赖具体模型或某些规格，应将贡献写成“识别评价解释的边界”，而不能写成适用于所有方法的历史长度定律。

{old_note}

## 整篮范围与辅助检查

完整模型的复购 Recall@10 为 {metrics.loc['full','recall_at_10']:.2%}，对整张下一单的 Recall@10 为 {scope['whole_basket_recall']:.2%}。全部历史候选对下一单的平均覆盖比例为 {scope['repeat_target_candidate_coverage']:.2%}；新品不在候选中。这些分母不同，不能统称购物篮“准确率”。预先指定的 K=5、20 结果保存在 `tables/k_sensitivity.csv`，未用于重新选择模型。

独立审核 `{audit['status']}`：排除 {audit['previous_users_excluded']:,} 名此前研究用户，旧新重叠为 {audit['previous_user_overlap']}；从原始 CSV 重建 {audit['confirmation_candidate_rows']:,} 个确认候选和 {audit['confirmation_target_rows']:,} 条目标记录；审核 {audit['prediction_configurations_checked']} 个预测配置的排序及四项逐用户指标。还检查训练专属邻居库，并对 {audit['quota_user_seed_checks']:,} 个用户×种子比较验证名额守恒，发现 {audit['quota_count_mismatches']} 个不一致。详见 `audit.json`。

## 如何用于课程项目

推荐把主线组织为“假设—诊断—设计—验证”：先说明基础方法为何需要领域基线，再解释局部收益与总体收益的区别，提出配额约束并诚实报告验证选择，最后展示分组评价如何受人群构成影响。保留负结果能使推理链完整；不能把没有采用的非零重排描述成成功算法。

仍有限制：只有 Instacart 一个平台；领域基线适配的是复购子任务；训练/确认划分虽然更换但仍非跨时期回测；没有真实在线购物实验；方法网格有界，并未穷举全部邻居、规则或重排设计。新增实验提供课程级证据，不构成全领域的新颖性或最佳性能证明。

研究方案、实现与证据稿由 Codex 辅助生成并实际执行。小组应理解和复核方法，独立组织最终报告、海报和答辩，并按课程规定披露 AIGC 用途。研究数据不随报告对外发布。
'''
    (out/"research_extension.md").write_text(main, encoding="utf-8")
    return {"report": str(out/"research_extension.md"), "figures": 4, "audit": audit["status"],
            "selected_quota_weight": weight, "history_models": list(history)}
