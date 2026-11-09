# Instacart Next-Basket Repurchase Lab

研究问题：个人购买习惯之外，用户分群和商品关联能否改善下一单复购预测？

这是课程项目的可运行研究起点。结论以本地实际实验为准，不预设聚类、关联规则或复杂模型一定有收益。

## 环境与运行

原始六个 CSV 放在 `archive/`。代码不修改原始数据，不将原始数据、虚拟环境或大型缓存提交到 Git。

标准环境（Python 3.11–3.13）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run.py prepare --sample-size 10000
.\.venv\Scripts\python.exe run.py enrich
```

本机已创建 `.venv`，复用已有 Anaconda 科学计算包，并在项目环境中安装 LightGBM 和 mlxtend；具体版本可通过 `python -m pip freeze` 查看。

首次准备后不要覆盖缓存。更换样本量、种子或特征协议时，指定全新的 `--cache` 路径，并将同一路径传入后续命令。

先提交代码到本地 Git，再运行有界实验循环：

```powershell
.\.venv\Scripts\python.exe scripts/autoresearch_loop.py
```

循环输出运行目录，并写入 `outputs/latest_run.txt`。冻结验证集选定模型后，只执行一次最终测试：

```powershell
$runPath = Get-Content outputs/latest_run.txt
.\.venv\Scripts\python.exe run.py finalize --run $runPath
.\.venv\Scripts\python.exe -m streamlit run app.py
```

单个配置的验证（不会读取测试标签）：

```powershell
.\.venv\Scripts\python.exe run.py verify --config configs/candidate.json --output outputs/manual_validation
```

## 数据与问题边界

- 数据为 Instacart Online Grocery Shopping Dataset 2017。原始数据：206,209 用户、3,421,083 订单、49,688 商品。
- 仅从最后一单有真实标签的 131,209 用户中，以固定随机种子 42 抽样 10,000 用户。按用户分为 7,000 / 1,500 / 1,500。
- 每位用户保留全部 prior 历史。以最后一张 `train` 订单为预测目标。原竞赛的 `test` 没有商品标签，不直接用于本地评价。
- 候选集是用户历史上买过的所有商品，没有负采样。标签由最后一单中是否出现该商品构造，不直接照搬 `reordered`。
- 预测时点设在最后一张已观察订单之后。下一单日期、星期、小时、订单间隔均不作为输入。
- 商品价格、消费金额和绝对日期缺失，因此不做金额分群或严格全平台日历回测。订单间隔在 30 天封顶。
- 冷启动用户、新品发现、忘买识别和推荐带来的销售因果增益均不在第一版范围内。

## 公平比较与数据隔离

全局商品统计、标准化、K-means 和 FP-Growth 只拟合训练用户的 prior 历史。验证及测试用户仅贡献自己的个人历史特征。目标订单不会参与任何历史特征计算。

测试标签单独保存在 `labels_test.parquet`。迭代命令只打开 train 和 valid 标签。整个优化循环冻结评价代码、测试代码、特征表和划分；只允许改变 `configs/candidate.json`。最终测试对事先声明的全部比较配置一次性评价，不能再据测试结果改选模型。

主要指标是每用户 Recall@10，目标集合仅包括复购商品。复购目标为空时 Recall/NDCG 未定义，单独统计并排除这两个指标的平均；Precision/F1 仍包含这些订单。候选少于 10 时返回全部，Precision 分母为实际推荐数量。排序同分时以商品 ID 升序确定。

同时输出 NDCG@10、Precision@10、F1@10、推断耗时，及按用户进行的配对 bootstrap 差值区间（1,000 次）。F1@10 是固定长度指标，不是原 Kaggle 可变长度 F1 的直接复现。

## 实验矩阵

`configs/experiments.json` 固定以下八组比较：个人频次、上一单优先、指数衰减频次、逻辑回归、LightGBM 基础、加聚类、加关联规则、两者同时加入。

第一版 K-means 固定 6 簇；规则在训练历史最常见的 500 个商品上挖掘二项集，支持度至少 0.002 且共现次数至少 30，规则 lift 至少 1.1。规则分数由上一张已观察购物篮触发。共购向未来购买的迁移是否有效是待检验假设。

关联规则的商品范围限制不改变最终候选集。未命中规则的候选保留并使用零规则特征。LightGBM 固定 180 棵树，所有消融共享训练参数；第一轮不做大规模调参。

后续可研究 K 值和聚类稳定性、分群规则、Apriori/FP-Growth 效率对比、不同样本规模、更多随机种子及可变长度清单。它们不是本轮已完成结果。

## Autoresearch 的使用

使用 [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch) 的 Codex skill 2.3.0，按经典循环实现：假设 → 单一配置修改 → 本地 Git 提交 → 验证集评价 → 测试保护 → 保留/撤销 → TSV 记录。

先构建研究基础设施，再固定 1 个初始基线和 7 次有界实验；关闭 fork，不无限运行，不发布或推送。原始数据只读。失败配置的实验记录与验证模型保留用于报告，但候选配置的 Git 提交会撤销。

每个运行目录保存：`config.json`、`experiment_plan.json`、冻结文件哈希、`loop-results.tsv`、验证日志、`selected.json`、`handoff.json`、测试表格、bootstrap 区间与图表。

注意：保留/撤销依据探索性的验证集均值，不代表差异具有统计显著性。测试集是独立报告，不作为第二轮优化信号。

## 文件职责

- `basketlab/data.py`：数据抽样、划分、历史特征。
- `basketlab/enrichment.py`：分群、规则及特征。
- `basketlab/models.py`：基线、逻辑回归、LightGBM。
- `basketlab/metrics.py`：统一评价与 bootstrap。
- `basketlab/experiment.py`：训练、冻结测试、图表。
- `scripts/autoresearch_loop.py`：实验提交、评价、保护与撤销。
- `tests/test_protocol.py`：时间边界、数据隔离和手算指标测试。
- `app.py`：只读本地购物清单演示。

## 来源与 AIGC 披露

- [下载副本](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)
- [原始发布说明](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2)
- [数据字典](https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b)
- [原始条款](https://gist.github.com/jeremystan/582eba13d6ee27ed465c43dc78934700)
- [A Next Basket Recommendation Reality Check](https://arxiv.org/abs/2109.14233)

当前研究方案、代码和测试由 Codex 辅助生成并在本机执行验证。小组应理解、复核和改写提交材料，核对引用，并按照课程要求披露 AIGC 用途。本 README 不是可直接提交的最终课程报告。
