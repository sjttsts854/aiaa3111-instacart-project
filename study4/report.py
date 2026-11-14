"""Render the completed frozen study without fitting or selecting anything.

All narrative numbers and figures are derived from saved confirmation outputs.
This presentation module is intentionally excluded from the research freeze.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICIES = ["full", "recent", "dual", "frequency_budget", "recency_budget"]
POLICY_ZH = {"full": "完整历史候选", "recent": "仅最近5单", "dual": "冻结选择的策略",
             "frequency_budget": "同预算：最高频", "recency_budget": "同预算：最近购买"}
CONTRASTS = {
    "frequency_information": ("频次：增加评分历史", "Frequency: more scoring history"),
    "decay_information": ("衰减：增加评分历史", "Decay: more scoring history"),
    "frequency_expansion": ("频次：扩展候选池", "Frequency: expand candidate pool"),
    "decay_expansion": ("衰减：扩展候选池", "Decay: expand candidate pool"),
    "frequency_interaction": ("频次：两因素交互", "Frequency: interaction"),
    "decay_interaction": ("衰减：两因素交互", "Decay: interaction"),
    "dual_vs_full": ("选定策略 − 完整候选", "Selected policy - full pool"),
    "dual_vs_frequency_budget": ("选定策略 − 同预算最高频", "Selected - frequency budget"),
    "dual_vs_recency_budget": ("选定策略 − 同预算最近购买", "Selected - recency budget"),
}
SPEC_LABELS = {
    "candidate_only_min10": "Candidate count; min 10",
    "candidate_only_min20": "Candidate count; min 20",
    "candidate_historical_basket_min10": "Count + basket size; min 10",
    "candidate_historical_basket_min20": "Count + basket size; min 20",
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _num(value, digits=2):
    return "未定义" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _pct(value):
    return "未定义" if value is None or not np.isfinite(value) else f"{100 * value:.2f}%"


def _pp(value, digits=2):
    return "未定义" if value is None or not np.isfinite(value) else f"{100 * value:+.{digits}f}"


def _ci(row, prefix="sim", digits=2):
    return f"[{_pp(row[f'{prefix}_low'], digits)}, {_pp(row[f'{prefix}_high'], digits)}]"


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def _save(fig, path):
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_history(raw, sensitivity, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7), gridspec_kw={"width_ratios": [0.85, 1.5]})
    x = np.arange(len(raw))
    axes[0].plot(x, 100 * raw.recall, "o-", color="#235789", lw=2)
    for pos, row in enumerate(raw.itertuples()):
        axes[0].annotate(f"n={int(row.nonempty_users):,}\nC={row.candidate_mean:.1f}",
                         (pos, 100 * row.recall), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set(xticks=x, xticklabels=raw.history_group, xlabel="Observed prior orders",
                ylabel="Macro Recall@10 (%)", title="A. Raw history-length pattern")
    axes[0].margins(y=0.3)
    y = np.arange(len(sensitivity))
    axes[1].axvline(0, color="#555555", lw=0.9)
    raw_delta = sensitivity.raw_overall_long_minus_short.iloc[0]
    if np.isfinite(raw_delta):
        axes[1].axvline(100 * raw_delta, color="#929292", ls=":", label="Raw overall")
    for offset, col, low, high, color, label in [
        (-0.13, "raw_common_support_long_minus_short", "raw_common_support_ci95_low", "raw_common_support_ci95_high", "#a56e2a", "Raw on common support"),
        (0.13, "standardized_long_minus_short", "standardized_ci95_low", "standardized_ci95_high", "#1a7d72", "Standardized on same users"),
    ]:
        values = sensitivity[col].to_numpy(float) * 100
        for pos, val, lo, hi in zip(y, values, sensitivity[low] * 100, sensitivity[high] * 100):
            if np.isfinite(val) and np.isfinite(lo) and np.isfinite(hi):
                axes[1].plot([lo, hi], [pos + offset] * 2, color=color, lw=2)
        axes[1].scatter(values, y + offset, color=color, s=32, label=label, zorder=4)
    labels = [f"{SPEC_LABELS[row.specification]}\nretained: {100*row.retained_fraction:.1f}%" for row in sensitivity.itertuples()]
    axes[1].set(yticks=y, yticklabels=labels, xlabel="16+ minus 3-5 orders (percentage points)",
                title="B. Separate support restriction from reweighting")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8, loc="best")
    fig.text(0.5, 0.0, "Descriptive, marginal 95% intervals; fixed support and weights. Recall excludes empty repeat targets. C = mean historical candidates.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    _save(fig, path)


def _plot_factorial(factorial, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.5))
    values = []
    part = factorial[factorial.subgroup.eq("long_history")].set_index("name")
    for method in ["frequency", "decay"]:
        values.append(np.array([[part.loc[f"{method}_s{s}_c{c}", "recall_at_10"] for c in ["5", "all"]]
                                for s in ["5", "all"]], dtype=float))
    low, high = np.nanmin(values), np.nanmax(values)
    for ax, method, matrix in zip(axes, ["Frequency score", "Decay score (0.8)"], values):
        ax.imshow(matrix, cmap="Blues", vmin=low, vmax=high)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{100*value:.2f}%", ha="center", va="center", fontsize=17,
                    color="white" if value > low + .6*(high-low) else "#172a3a")
        ax.set(xticks=[0, 1], xticklabels=["Recent 5", "All history"], yticks=[0, 1],
               yticklabels=["Recent 5", "All history"], xlabel="Eligible candidate pool", ylabel="History used for scoring", title=method)
    fig.suptitle("Same users, same targets: two distinct uses of history", fontsize=13)
    fig.text(.5, .01, "Users with 16+ prior orders; Macro Recall@10. All cells know the complete historical catalog; excluded targets remain misses.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    _save(fig, path)


def _plot_contrasts(primary, margin, path):
    part = primary.set_index("contrast").loc[list(CONTRASTS)].reset_index()
    fig, ax = plt.subplots(figsize=(11.8, 5.7))
    for pos, row in enumerate(part.itertuples()):
        color = "#1a7d72" if row.sim_low > 0 else ("#b65339" if row.sim_high < 0 else "#52677c")
        ax.plot([100*row.sim_low, 100*row.sim_high], [pos]*2, color=color, lw=2.2)
        ax.scatter(100*row.delta, pos, color=color, s=33, zorder=3)
    ax.axvline(0, color="#555555", lw=1)
    # The margin applies to the selected-policy/full-pool noninferiority claim.
    ax.axvline(-100*margin, color="#929292", ls="--", lw=1, label="Noninferiority margin (selected vs full only)")
    ax.axhline(5.5, color="#dddddd", lw=1)
    ax.set(yticks=np.arange(len(part)), yticklabels=[CONTRASTS[c][1] for c in part.contrast],
           xlabel="Paired difference in Macro Recall@10 (percentage points)",
           title="Nine prespecified contrasts: simultaneous 95% intervals")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="best")
    fig.text(.5, .005, "First six: 16+ prior orders. Last three: all users with nonempty repeat targets. Bonferroni percentile bootstrap, family = 9.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    _save(fig, path)


def _plot_ledger(ledger, path):
    part = ledger.loc[ledger.subgroup.eq("long_history")].copy()
    part["kind"] = part.name.str.replace(r"_\d+$", "", regex=True)
    avg = part.groupby("kind")[["new_true_recall", "lost_old_true_recall", "delta_recall"]].mean()
    order = [name for name in ["frequency", "decay", "learned_full", "learned_dual"] if name in avg.index]
    avg = avg.loc[order]
    x = np.arange(len(avg))
    fig, ax = plt.subplots(figsize=(10.5, 4.7))
    ax.bar(x-.16, 100*avg.new_true_recall, width=.3, color="#1a7d72", label="New candidate true hits / full targets")
    ax.bar(x+.16, -100*avg.lost_old_true_recall, width=.3, color="#b65339", label="Displaced old true hits / full targets")
    ax.scatter(x, 100*avg.delta_recall, color="#172a3a", marker="D", s=45, label="Net Recall change", zorder=4)
    for pos, row in enumerate(avg.itertuples()):
        ax.annotate(f"{100*row.delta_recall:+.2f}", (pos, 100*row.delta_recall), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(0, color="#666666", lw=.9)
    labels = {"frequency": "Frequency\nto full pool", "decay": "Decay\nto full pool", "learned_full": "Learned score\nto full pool", "learned_dual": "Learned score\nto selected pool"}
    ax.set(xticks=x, xticklabels=[labels[c] for c in order], ylabel="Recall contribution (percentage points)",
           title="Expanding recent-5 candidates: recovered hits versus displaced hits")
    ax.legend(fontsize=8, loc="best")
    fig.text(.5, .005, "16+ prior orders; identical scores within each expansion. Exact user-level identity, then macro mean. Learned rows average three model seeds.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    _save(fig, path)


def _plot_budget(curves, metrics, product_groups, path):
    avg = curves.groupby(["policy", "old_count"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {"dual": "#1a7d72", "frequency_budget": "#a56e2a", "recency_budget": "#52677c"}
    names = {"dual": "Recent + old frequent", "frequency_budget": "Frequency budget", "recency_budget": "Recency budget"}
    for policy, color in colors.items():
        part = avg.loc[avg.policy.eq(policy)].sort_values("mean_candidates")
        axes[0].plot(part.mean_candidates, 100*part.recall_at_10, "o-", ms=4, color=color, label=names[policy])
        axes[1].plot(part.mean_candidates, 100*part.candidate_coverage, "o-", ms=4, color=color)
    full, chosen = metrics.loc["full"], metrics.loc["dual"]
    for ax, column in zip(axes[:2], ["recall_at_10", "candidate_coverage"]):
        ax.scatter(full.mean_candidates, 100*full[column], marker="s", color="#222222", s=42, label="Full pool", zorder=4)
        ax.scatter(chosen.mean_candidates, 100*chosen[column], marker="*", color="#b65339", s=155, label="Frozen selected policy", zorder=5)
        ax.set_xlabel("Mean eligible candidates per user")
    axes[0].set(ylabel="Macro Recall@10 (%)", title="A. Candidate budget and ranking quality")
    axes[1].set(ylabel="Full repeat-target coverage (%)", title="B. Targets remaining eligible")
    axes[0].legend(fontsize=7.7, loc="best")
    group_order = ["recent", "dormant_repeat", "dormant_low_frequency"]
    grouped = product_groups.groupby(["policy", "product_group"]).mean(numeric_only=True)
    x = np.arange(3)
    for offset, policy, color, label in [(-.24, "full", "#52677c", "Full"), (0, "recent", "#a56e2a", "Recent only"), (.24, "dual", "#1a7d72", "Selected")]:
        vals = grouped.loc[policy].reindex(group_order).micro_recall
        axes[2].bar(x+offset, 100*vals, width=.23, color=color, label=label)
    axes[2].set(xticks=x, xticklabels=["Recent", "Old, count >=3", "Old, count <3"], ylabel="Micro recall within product group (%)",
                title="C. What is lost when we forget?")
    axes[2].tick_params(axis="x", labelsize=8)
    axes[2].legend(fontsize=8)
    fig.text(.5, .005, "Confirmation curves are descriptive only: policy was selected on validation. Candidate-count reduction is not a measured runtime speedup.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    _save(fig, path)


def report(cache, run):
    """Write reports/study4/research_story.md, five figures and source tables.

    Require completed confirmation. An audit can still be pending; rerunning
    after it passes refreshes the report without touching frozen experiment data.
    """
    cache, run = Path(cache), Path(run)
    dest = run / "confirmation"
    if not (dest / "completed.json").exists():
        raise RuntimeError("Confirmation must be complete before generating research findings")
    frozen = _read(run / "freeze.json")
    config, selection = frozen["protocol"], _read(run / "selection.json")
    audit = _read(run / "audit.json") if (run / "audit.json").exists() else {"status": "PENDING"}
    if audit.get("status") not in {"PASS", "PENDING"}:
        raise RuntimeError("Cannot present findings as verified after an audit failure")
    primary = pd.read_csv(dest / "primary_contrasts.csv")
    if primary.contrast.duplicated().any() or set(primary.contrast) != set(CONTRASTS):
        raise ValueError("The report requires all nine prespecified primary contrasts")
    if len(config["primary_contrasts"]) != 9:
        raise ValueError("Simultaneous interval family changed")
    metrics = pd.read_csv(dest / "metrics.csv").set_index("name")
    validation = pd.read_csv(run / "policy_validation.csv")
    factorial = pd.read_csv(dest / "factorial_metrics.csv")
    ledger = pd.read_csv(dest / "ledger_summary.csv")
    curves = pd.read_csv(dest / "candidate_curves.csv")
    products = pd.read_csv(dest / "product_group_diagnostics.csv")
    scope = pd.read_csv(dest / "task_scope.csv").set_index("policy")
    history_dir = dest / "history_observed"
    raw = pd.read_csv(history_dir / "raw_overall.csv")
    sensitivity = pd.read_csv(history_dir / "sensitivity.csv")
    cells = pd.read_csv(history_dir / "cells.csv")
    history_summary = _read(history_dir / "summary.json")
    k_sensitivity = pd.read_csv(dest / "k_sensitivity.csv")
    splits = pd.read_csv(cache / "splits.csv").groupby("split").user_id.nunique()
    out = Path("reports/study4")
    figures, tables = out / "figures", out / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .17, "axes.axisbelow": True})
    _plot_history(raw, sensitivity, figures / "01_observed_history.png")
    _plot_factorial(factorial, figures / "02_factorial_memory.png")
    _plot_contrasts(primary, config["practical_margin"], figures / "03_primary_contrasts.png")
    _plot_ledger(ledger, figures / "04_expansion_ledger.png")
    _plot_budget(curves, metrics, products, figures / "05_budget_tradeoff.png")

    sources = {
        "policy_validation.csv": run / "policy_validation.csv",
        "confirmation_metrics.csv": dest / "metrics.csv",
        "primary_contrasts.csv": dest / "primary_contrasts.csv",
        "factorial_metrics.csv": dest / "factorial_metrics.csv",
        "factorial_sensitivity.csv": dest / "factorial_sensitivity.csv",
        "ledger_summary.csv": dest / "ledger_summary.csv",
        "candidate_curves.csv": dest / "candidate_curves.csv",
        "product_group_diagnostics.csv": dest / "product_group_diagnostics.csv",
        "task_scope.csv": dest / "task_scope.csv",
        "k_sensitivity.csv": dest / "k_sensitivity.csv",
        "history_raw_overall.csv": history_dir / "raw_overall.csv",
        "history_sensitivity.csv": history_dir / "sensitivity.csv",
        "history_groups.csv": history_dir / "groups.csv",
        "history_cells.csv": history_dir / "cells.csv",
    }
    for filename, source in sources.items():
        shutil.copyfile(source, tables / filename)
    _write_json(out / "protocol.json", config)
    _write_json(out / "selection.json", selection)
    _write_json(out / "audit_summary.json", {key: value for key, value in audit.items() if key != "independent_metrics"})
    _write_json(out / "history_summary.json", history_summary)
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run).replace("\\", "/"), "cache": str(cache).replace("\\", "/"),
        "freeze_sha256": hashlib.sha256((run / "freeze.json").read_bytes()).hexdigest(),
        "freeze_utc": frozen["utc"], "selection_utc": selection["utc"],
        "confirmation_started": _read(dest / "started.json"), "confirmation_completed": _read(dest / "completed.json"),
        "audit_status": audit["status"], "frozen_file_count": len(frozen["sha256"]),
        "table_sha256": {filename: hashlib.sha256(source.read_bytes()).hexdigest() for filename, source in sources.items()},
        "table_sources": {filename: str(source).replace("\\", "/") for filename, source in sources.items()},
        "seed_aggregation": "Metrics averaged within users across three fitted model seeds; not a prediction ensemble.",
        "report_does_not_fit_select_or_rerun_confirmation": True,
    }
    _write_json(out / "provenance.json", provenance)

    pc = primary.set_index("contrast")
    full, chosen = metrics.loc["full"], metrics.loc["dual"]
    margin = config["practical_margin"]
    ni = bool(pc.loc["dual_vs_full", "sim_low"] >= -margin)
    reduction = 1 - chosen.mean_candidates / full.mean_candidates
    policy = selection["policy"]
    policy_name = f"最近5单商品 + 最多 {policy['m']} 个历史高频旧商品" if policy["kind"] == "dual" else "完整历史候选（验证阶段回退）"
    main_delta = pc.loc["dual_vs_full"]
    if policy["kind"] == "all":
        practical = "验证阶段未选出满足条件的压缩策略，冻结方案回退到完整候选；因此本轮没有建立候选压缩收益。"
    elif ni:
        practical = (f"冻结方案“{policy_name}”把平均候选数从 {full.mean_candidates:.2f} 降至 {chosen.mean_candidates:.2f}"
                     f"（减少 {_pct(reduction)}），独立确认满足预设 0.5 个百分点的 Recall@10 非劣界限。")
    else:
        practical = (f"冻结方案“{policy_name}”虽将平均候选数减少 {_pct(reduction)}，但独立确认未通过预设 0.5 个百分点的非劣检验。"
                     "不能把它描述为已经确认的‘无损压缩’。")
    history_delta = float(sensitivity.raw_overall_long_minus_short.iloc[0])
    observed_line = (f"确认集里，16+ 单用户相对 3–5 单用户的原始 Recall@10 差为 {_pp(history_delta)} 个百分点。"
                     + ("“更多订单对应更难预测”的起点在本样本成立。" if history_delta < 0 else "本样本没有复现“更多订单更难预测”的原始方向，故事需保留为研究问题。"))
    count_positive = int(sensitivity.standardized_long_minus_short.gt(0).sum())
    count_negative = int(sensitivity.standardized_long_minus_short.lt(0).sum())
    observed_interpretation = (f"四种预设方案中，{int(sensitivity.standardized_long_minus_short.notna().sum())} 种有可比支持；"
                              f"标准化点差 {count_positive} 种为正、{count_negative} 种为负。")
    if history_summary["all_four_reversed_within_common_support"]:
        observed_interpretation += "四种方案均在同一共同支持样本内发生方向反转，说明原始组间差异对历史候选构成的权重敏感。"
    elif history_summary["all_four_reversed_from_overall"]:
        observed_interpretation += "四种标准化点差均与全体原始差异反向，但不能把方向变化全部归给重加权：还包含共同支持筛选。"
    else:
        observed_interpretation += "结果不支持把‘标准化后稳定反转’预设为故事结论；应逐项报告共同支持及加权变化。"
    support_signatures = set()
    for _, group in cells.groupby("specification"):
        cols = ["history_group", "candidate_bin", "historical_basket_bin", "users", "pooled_weight"]
        support_signatures.add(group[cols].sort_values(cols[:3]).to_csv(index=False))
    support_note = (f"四种预设规格在本数据产生 **{len(support_signatures)} 套不同的支持单元与权重**。"
                    + ("同一分层方案的10/20人门槛没有改变支持，不能称为四次独立复制。" if len(support_signatures) == 2 else
                       "这些是同一确认样本的敏感性检查，不是独立复制。"))
    fi, di = pc.loc["frequency_information"], pc.loc["decay_information"]
    fe, de = pc.loc["frequency_expansion"], pc.loc["decay_expansion"]
    mechanism = (f"在完整候选池上，增加评分历史对频次的影响为 **{_pp(fi.delta)}** 个百分点，对衰减的影响为 **{_pp(di.delta)}**；"
                 f"保持全历史分数固定后扩展候选，对频次的影响为 **{_pp(fe.delta)}**，对衰减为 **{_pp(de.delta)}**。")
    if fi.sim_high < 0 and di.sim_low > 0 and fe.sim_high < 0 and de.sim_low > 0:
        mechanism += "这四个方向均由同时区间确认：历史信息和旧商品并非天然有害，其作用取决于评分是否处理了购买时间。不能由频次模型的退化推出所有模型都应该忘掉旧历史。"
    else:
        mechanism += "方向是否成立应逐项查看同时区间；不同评分的结果不能替代彼此。"
    actual_scope = scope.loc["full"]
    summary_rows = [[POLICY_ZH[name], _pct(metrics.loc[name, "recall_at_10"]),
                     _pct(metrics.loc[name, "ndcg_at_10"]), _num(metrics.loc[name, "mean_candidates"]),
                     _pct(metrics.loc[name, "candidate_coverage"]), _pct(metrics.loc[name, "oracle_recall_at_10"])] for name in POLICIES]
    contrast_rows = []
    for name in CONTRASTS:
        row = pc.loc[name]
        direction = "正向差异已确认" if row.sim_low > 0 else ("负向差异已确认" if row.sim_high < 0 else "区间跨零，方向未确认")
        if name == "dual_vs_full":
            direction += "；非劣" + ("通过" if ni else "未通过")
        contrast_rows.append([CONTRASTS[name][0], int(row.users), _pp(row.delta), _ci(row), direction])
    history_rows = [[SPEC_LABELS[row.specification], int(row.retained_users), _pct(row.retained_fraction),
                     _pp(row.raw_overall_long_minus_short), _pp(row.raw_common_support_long_minus_short),
                     _pp(row.standardized_long_minus_short),
                     f"[{_pp(row.standardized_ci95_low)}, {_pp(row.standardized_ci95_high)}]"] for row in sensitivity.itertuples()]
    group_rows = [[row.history_group, int(row.users), int(row.nonempty_users), _num(row.candidate_mean), _pct(row.recall)] for row in raw.itertuples()]
    factorial_rows = []
    lf = factorial[factorial.subgroup.eq("long_history")].set_index("name")
    for score_window, pool in [("5", "5"), ("all", "5"), ("5", "all"), ("all", "all")]:
        factorial_rows.append(["最近5单" if score_window == "5" else "全部历史", "最近5单" if pool == "5" else "全部历史",
                               *[_pct(lf.loc[f"{method}_s{score_window}_c{pool}", "recall_at_10"]) for method in ["frequency", "decay"]]])
    ledger_part = ledger[ledger.subgroup.eq("long_history")].copy()
    ledger_part["kind"] = ledger_part.name.str.replace(r"_\d+$", "", regex=True)
    ledger_avg = ledger_part.groupby("kind").mean(numeric_only=True)
    ledger_rows = [[label, _pp(ledger_avg.loc[name, "new_true_recall"]), _pp(-ledger_avg.loc[name, "lost_old_true_recall"]),
                    _pp(ledger_avg.loc[name, "delta_recall"])] for name, label in [("frequency", "频次：最近→完整"),
                    ("decay", "衰减：最近→完整"), ("learned_full", "学习模型：最近→完整"), ("learned_dual", "学习模型：最近→冻结方案")]]
    product_avg = products.groupby(["policy", "product_group"]).mean(numeric_only=True)
    product_rows = []
    for name, label in [("recent", "最近5单买过"), ("dormant_repeat", "旧商品且累计购买≥3次"), ("dormant_low_frequency", "旧商品且累计购买<3次")]:
        if ("full", name) in product_avg.index:
            pr = product_avg.loc[("full", name)]
            product_rows.append([label, _pct(pr.target_share), _pct(pr.micro_recall),
                                 _pct(product_avg.loc[("recent", name), "micro_recall"]), _pct(product_avg.loc[("dual", name), "micro_recall"])])
    k_lines = []
    for k, part in k_sensitivity.groupby("k"):
        mean = part.groupby("policy").mean(numeric_only=True)
        k_lines.append(f"K={int(k)} 时冻结方案相对完整候选的 Recall 点差为 {_pp(mean.loc['dual', f'recall_at_{int(k)}']-mean.loc['full', f'recall_at_{int(k)}'], 4)} 个百分点")
    twenty = k_sensitivity.loc[k_sensitivity.k.eq(20)].groupby("policy").mean(numeric_only=True)
    k_boundary = ""
    if not twenty.empty and twenty.loc["dual", "recall_at_20"] < twenty.loc["full", "recall_at_20"]:
        k_boundary = (f"K=20 的点估计损失比主要 K=10 结果明显：完整候选 {_pct(twenty.loc['full', 'recall_at_20'])}，"
                      f"冻结方案 {_pct(twenty.loc['dual', 'recall_at_20'])}。这是适用范围的反例，不能把 Top10 的非劣结论直接外推到更长列表；这里没有另作确认性区间检验。")
    budget_successes = [CONTRASTS[name][0] for name in ["dual_vs_frequency_budget", "dual_vs_recency_budget"] if pc.loc[name, "sim_low"] > 0]
    budget_statement = ("两个同预算基线均被同时区间确认超过，说明收益不限于候选数减少。" if len(budget_successes) == 2 else
                        f"两个同预算对比中，{len(budget_successes)} 个通过同时区间确认正向差异；不能宣称对所有同预算选择都更优。")
    audit_text = (f"独立审计 **PASS**：检查了 {audit.get('confirmation_users', int(full.users)):,} 名确认用户的原始时序、标签、完整候选、冻结哈希、排名及指标。"
                  if audit["status"] == "PASS" else "独立审计 **PENDING**：以下数值已由完成的确认流程产出，尚待独立审计；不能标注为已审计通过。")
    report_text = f"""# 订单越多，下一单反而更难预测？购物推荐应该忘掉什么

本轮把原来的问题改写成一个可以被实验推翻的命题：**历史既用于给商品评分，也决定哪些商品有资格竞争前10名。两种作用必须分开检验。**

{observed_line} {practical} 选定方案相对完整候选的 Recall@10 差为 **{_pp(main_delta.delta, 4)} 个百分点**，九项主要对比的同时95%区间为 **{_ci(main_delta, digits=4)}**。本轮主线落为：**区分评分信息和候选资格，检验怎样利用时间信息、怎样在明确损失边界内精简候选，并保留覆盖代价与反例。**

## 1. 任务和证据边界

使用 Instacart 已观察历史预测用户终末一单中再次购买的商品。训练、验证、确认用户各为 **{int(splits.get('train', 0)):,} / {int(splits.get('valid', 0)):,} / {int(splits.get('confirm', 0)):,}**；本轮排除此前实验已使用的用户。完整历史重复购买目标记为 R_full，任何候选删减都不改变 R_full、不删除用户，也不把被排除的真目标变成“无需预测”。没有目标的用户仍计数，但其 Recall/NDCG 不定义。

主要指标是用户平均 Recall@10，不是分类准确率。完整候选模型的重复购买 Recall 为 **{_pct(actual_scope.macro_repeat_recall)}**；按整篮全部商品作分母只有 **{_pct(actual_scope.macro_whole_basket_recall)}**。重复商品占整篮目标总数 **{_pct(actual_scope.repeat_target_share_micro)}**，所以本项目不声称解决首次购买商品推荐。

学习模型沿用 LightGBM 的既有特征与训练设置，三个随机种子分别拟合。先对每名用户的指标取三个种子的均值，再做配对分析；**这不是预测分数集成**。验证集只选择旧商品保留数 M，随后冻结协议、数据与模型，再一次性打开新确认集。所有九项主要对比使用 {config['bootstrap_replicates']:,} 次用户配对 bootstrap，并报告 Bonferroni 同时95%区间；不把多个种子当作独立用户扩增样本量。

{audit_text} 追踪信息见 [provenance.json](provenance.json)、[协议](protocol.json)、[审计摘要](audit_summary.json)。报告重跑只更新展示文件，不重新训练或选择策略。

## 2. A：先检验“历史越多越难”的观察

{_table(['历史订单数', '全部用户', '非空目标用户', '平均历史候选数', 'Recall@10'], group_rows)}

![原始历史趋势与共同支持标准化](figures/01_observed_history.png)

只使用预测时可见的历史候选数、历史平均篮子大小分层，分别要求每个历史组在每格至少10或20人，四种方案全部呈现。下表的差异均为“16+单 − 3–5单”，单位为百分点。

{_table(['分层方案', '保留用户', '保留比例', '全体原始差', '同支持原始差', '标准化差', '标准化边际95%区间'], history_rows)}

{observed_interpretation} {support_note} 全体原始→同支持原始是**样本筛选变化**；同支持原始→标准化才是**同一保留人群上的重加权变化**。两部分不能混称为“控制变量解释了下降”。标准化区间是四种敏感性诊断的边际区间，条件于已选支持、固定权重和既有模型，不能冒充九项主要对比的同时区间。

分层没有使用下一单商品数，但 Recall 人群仍受“未来重复目标非空”的纳入条件限制。粗分箱残留差异、未观测因素以及历史变量本身可能随历史积累变化，因此 A 只能说明组间比较对构成的敏感性，不能证明多购物导致更难或更易预测。另存的未来目标大小条件化分析仅作回顾敏感性，不承担主线证据。

## 3. B：把“记住多少”拆成两个因素

对相同的长历史用户（≥16单），固定同一未来目标和完整历史目录，只改变：①评分读取最近5单还是全部历史；②允许最近5单商品还是全部历史商品竞争。两种评分分别是购买频次、0.8衰减频次。最近评分仍知道完整目录，只是窗口外购买记录不计入分数。

{_table(['评分历史', '候选池', '频次 Recall@10', '衰减 Recall@10'], factorial_rows)}

![两因素实验](figures/02_factorial_memory.png)

“增加评分历史”是在完整候选池上比较全历史评分与最近5单评分；“扩展候选池”是在完全相同的全历史分数上比较完整候选与最近5单候选；交互项为前一个评分历史差异在两种候选池下的差中之差。它们回答不同问题，不能把窗口效果全部解释为行为遗忘。

{mechanism}

交互项还有一个设计上的边界：最近5单评分给所有窗口外商品零分。只要近期候选不少于10个，增加这些零分旧商品就不会改变前10名；近期候选不足10个时才可能补入。因而交互并非完全出乎意料的现象，它部分反映旧记录让旧商品重新进入排名。主解释优先依赖**全历史分数固定时的候选扩展及精确记账**，以及学习模型的冻结方案与同预算对照。

{_table(['主要对比', '非空目标用户', 'Recall差（百分点）', '同时95%区间', '证据判断'], contrast_rows)}

![九个预设主要对比的同时区间](figures/03_primary_contrasts.png)

扩展候选池可能找回旧商品，也可能让旧候选挤掉原来正确的推荐。逐用户核对的恒等式为：**新增候选带来的正确命中 − 被挤掉的原候选正确命中 = 总命中变化**。每一项除以同一个 R_full，再取用户均值，得到下表；单位为 Recall 百分点。它是离线排名的精确记账，不是关于消费者心理的因果分解。

{_table(['最近5单候选扩展', '找回真命中贡献', '挤出真命中贡献', '净变化'], ledger_rows)}

![候选扩展中的找回与挤出](figures/04_expansion_ledger.png)

本轮把“增加评分历史”和“扩大候选竞争”分开后，即使其中一个对比为负、另一个为正，也不矛盾。负向结果与方向尚未确认的结果同样用于界定哪种记忆值得保留。

## 4. C：能否保留信息、适度压缩候选？

候选方案为最近5单商品并集最多 M 个旧商品，旧商品按历史频次、最近购买距离、商品ID稳定排序，全部在观察历史内完成。验证集考察 M={config['dual_old_counts']}；先筛选相对完整候选的配对边际95%下限≥−0.005的方案，再选择平均候选数最少者；没有合格方案则回退完整候选。这个搜索仅是开发阶段选择，**不能当作确认结论**。

冻结方案为 **{policy_name}**。确认阶段非劣判定必须用九项对比的同时区间下限≥−0.005；本轮结果是 **{'通过' if ni else '未通过'}**。非劣只容许预设范围内的损失，不等于相等、不等于更好，也不等于完全不丢目标。

{_table(['候选策略', 'Recall@10', 'NDCG@10', '平均候选数', '目标覆盖率', '候选内Oracle Recall@10'], summary_rows)}

同预算基线对每名用户严格使用与冻结方案相同的候选数量，分别选择历史最高频或最近购买商品，再用同一学习模型分数排名。{budget_statement}

![候选预算与目标覆盖](figures/05_budget_tradeoff.png)

下面的商品分组仅用历史定义；“旧”指最近5单未买。目标占比和组内召回为按商品汇总的 micro 指标，与主要的用户平均 Recall 不同，不能直接混算。

{_table(['历史商品组', '目标占比', '完整候选组内召回', '仅最近组内召回', '冻结方案组内召回'], product_rows)}

覆盖率是全部重复目标中仍有资格被推荐的比例，再对用户平均；Oracle 还考虑最多只能返回10件。冻结方案只覆盖 **{_pct(chosen.candidate_coverage)}** 的完整重复目标，平均 **{_pct(1-chosen.candidate_coverage)}** 的目标在该策略下失去资格。即使Top10非劣成立也不代表目标覆盖无损；因为原模型本来未在前10名找回所有这些目标。低命中率也不能证明这些旧商品对用户不重要。删候选能减少竞争，却也制造不可恢复的漏召回。

本报告的 **{_pct(reduction)}** 减少按候选总量比计算，等于相同用户数下平均候选数的比；逐用户先求保留比例再平均则为 **{_pct(chosen.candidate_retention)}**，两种汇总不可混用。候选数下降只是潜在计算量的代理：本轮保留完整历史特征，先计算完整候选分数再筛选，**没有证明真实评分加速、历史存储节省或相同百分比的端到端提速**。

敏感性结果：{'；'.join(k_lines)}。{k_boundary} 窗口3/10的两因素结果、全部确认预算曲线和K敏感性均为已冻结方案之外的描述检查，不据此重新选择M或宣布新的主要胜利。

## 5. 可以答辩的贡献，以及明确的反例边界

1. **评估设计贡献**：把评分信息和候选资格拆开，固定用户、未来目标及 Recall 分母，避免通过删掉难预测目标制造收益。原始→同支持→标准化的三步同时防止把样本筛选误说成解释。
2. **机制证据贡献**：两种评分的2×2对照配合逐用户找回/挤出记账，说明“利用更多历史”和“让更多旧商品竞争”是可分离的算法操作。主要区间跨零或方向不一致时，结论必须相应收窄。
3. **可复现策略检验贡献**：把历史定义的双层候选方案与两个逐用户同预算基线放到新用户确认集，公开冻结过程、同时区间、覆盖损失及失败条件。{practical}

这些是课程项目层面的完整研究贡献，不宣称发明新推荐算法或达到最新最好水平。此前V2/V3仅提供提出本轮问题的探索背景，不能与新确认样本拼接扩充证据。

限制包括：单一数据集与离线终末订单任务；仅预测历史见过商品；历史长短是不同用户的观察比较；标准化仅在共同支持内有效；置信区间条件于已训练模型与支持，未包含重新采样训练全流程的不确定性；3/10窗口、K及商品组结果属于描述敏感性；完整目录在所有评分窗口中已知，不能推广为没有旧目录知识的冷启动。若非劣未通过，应明确保留失败结论，而不是依据确认曲线换M。

## 6. 建议的课堂讲述顺序

1. 用原始历史组曲线提出反直觉问题，同时标出候选数；先说这是观察而非因果。
2. 展示同支持与标准化，说明“用户构成”检查为什么必须拆开报告。
3. 用2×2图讲清评分记忆和候选记忆，再展示九项同时区间及找回/挤出记账。
4. 用冻结策略与同预算对照回答是否有实际价值，明确非劣是否通过及覆盖代价。
5. 用三个课程贡献与限制收尾：结论应以真实证据允许的范围为准，而不是‘历史越少越好’。

复现入口：`python -m study4.run report`。精选数表位于 [tables/primary_contrasts.csv](tables/primary_contrasts.csv)、[tables/history_sensitivity.csv](tables/history_sensitivity.csv)、[tables/confirmation_metrics.csv](tables/confirmation_metrics.csv)；完整源文件映射及哈希见 [provenance.json](provenance.json)。
"""
    (out / "research_story.md").write_text(report_text, encoding="utf-8")
    return {"report": str(out / "research_story.md"), "figures": 5, "source_tables": len(sources),
            "audit": audit["status"], "noninferiority_confirmed": ni, "selected_policy": policy,
            "candidate_reduction_ratio": float(reduction)}
