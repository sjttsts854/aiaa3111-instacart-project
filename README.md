# Instacart Next-Basket Repurchase Lab

研究问题：个人购买习惯之外，用户分群和商品关联能否改善下一单复购预测？

项目已从初始方法比较推进到固定假设、全新用户确认的解释性研究。结论以本地实际实验为准，不预设聚类、关联规则或复杂模型一定有收益。

## 第三轮补充研究（2026-11-14）

最新证据见 [领域基线、历史长度稳健性与名额约束实验](reports/study3/research_extension.md)。第二轮证据保留供追溯，解释以新报告中的复现与修正为准。

本轮排除此前 43,000 人，用全新 33,000 人重新划分为训练 20,000 / 验证 5,000 / 确认 8,000。补充作者代码语义的 TIFU-KNN（24 个有界验证配置，包含论文 Instacart 参数），训练用户专属邻居库，统一历史候选；与同时间权重的个人历史对照、公平训练的 LightGBM 比较。实现与论文文本存在已记录的差异，不能把本任务指标与论文整篮分数直接比较。来源及细节见 `study3/tifu_sources.md`。

历史长度诊断使用预定的 3 套分箱 × 2 个最低人数阈值，并在完整模型、个人频次和 TIFU 上检查；分别报告全体原始差异、同一重叠人群原始差异、统一权重后的差异及条件 bootstrap 区间。这避免把样本排除的影响都归给重新加权。

特色方法尝试为覆盖名额约束重排：固定原模型清单中的规则覆盖/未覆盖数量，在两组内重新排序。验证权重 0/0.25/0.5/1，选中 0，即不启用改动。因此保留失败实验，不把确认集与基础模型恒等的结果包装成新方法成功。它也不能独立证明所有非零权重都无效。

结果要点：TIFU 邻居信息相对其个人历史对照提升 0.61 个百分点，完整特征模型再高 1.96 个百分点；历史长度的标准化方向在新数据六套设置下稳定；第二轮规则覆盖内的小幅收益未稳定复现。新结果进一步限定了贡献：群体信息的增量价值需要连同已有个人基线一起讨论；TIFU 与规则使用的个人基线强弱不同，不能将差异仅归因于表示方式。历史长度分组指标必须检查比较总体。

本机查看/核验：

```powershell
.\.venv\Scripts\python.exe -m study3.run audit
.\.venv\Scripts\python.exe -m streamlit run app.py
```

从干净副本复现：先完成旧两轮的 `prepare` 生成排除用户清单，再依次执行下列命令。旧轮实验无需重新训练；第二轮准备需按原方案生成 33,000 用户划分。

```powershell
.\.venv\Scripts\python.exe -m study3.prepare
.\.venv\Scripts\python.exe -m pytest -q tests study3_tests
.\.venv\Scripts\python.exe -m study3.run develop
.\.venv\Scripts\python.exe -m study3.run freeze
.\.venv\Scripts\python.exe -m study3.run confirm
.\.venv\Scripts\python.exe -m study3.run audit
.\.venv\Scripts\python.exe -m study3.run report
```

缓存 `data/processed/study3/`；完整运行记录 `autoresearch/research-v3/`；小型证据与图 `reports/study3/`。代码、规则、模型、数据及选择在确认前冻结；已完成确认不允许覆盖。冻结文件之后不得改写用于复现旧结果的源码。报告代码与展示可单独更新。学习模型取三个种子的指标均值，TIFU 为确定性方法。运行研究无需为原始数据或模型建立公开仓库。

## 第二轮：信息增益与边界研究

当前主要结果见 [研究证据稿](reports/research_findings.md)，完整数值在 `reports/tables/`，可直接用于展示的图表在 `reports/figures/`（PNG 与 SVG）。第一轮结果保留在本文后半部，用于说明假设的来源，不与第二轮作为独立样本混用。

三个问题：近期行为与购买节奏各有多少条件增益；扩大共购规则和跨订单规则覆盖后是否有用；长历史用户的表现差异与候选规模、目标规模有什么关系。研究不以获得正结果为完成条件。

第二轮排除全部第一轮用户，重新抽取 33,000 人，分为训练 20,000、验证 5,000、确认 8,000。17 种预先声明的方法共 41 个拟合/基线配置；每个学习方法重复三个种子，参数共享，不做逐模型调参。选择依据验证集三个种子的平均 Recall@10。最终确认前冻结 78 个输入、模型与评估文件。确认后不再训练或改选。

预先指定 8 个主要配对比较、0.5 个百分点的实际意义阈值、6,000 次用户 bootstrap，同时报告多重比较调整区间。覆盖条件与历史分箱为探索性诊断；不宣称因果。所有学习模型表格值为三个模型的指标均值，不是集成模型。

本机已经运行的缓存不能覆盖。只查看证据、重新独立核验和开启演示：

```powershell
.\.venv\Scripts\python.exe -m research.run audit
.\.venv\Scripts\python.exe -m streamlit run app.py
```

从头复现第二轮时，在干净副本中放入六个原始 CSV，先按下方步骤准备第一轮 10,000 人划分（只需 `run.py prepare`，用于排除已经观察的用户），再依次运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m research.run prepare
.\.venv\Scripts\python.exe -m research.run features
.\.venv\Scripts\python.exe -m research.run develop
.\.venv\Scripts\python.exe -m research.run freeze
.\.venv\Scripts\python.exe -m research.run confirm
.\.venv\Scripts\python.exe -m research.run audit
.\.venv\Scripts\python.exe -m research.exploratory
.\.venv\Scripts\python.exe -m research.run report
```

`freeze` 需要本地 Git 仓库和至少一次提交，用于记录来源版本。`prepare/features/freeze/confirm` 保护已有完成结果，拒绝覆盖；`develop` 仅在输入哈希一致且未冻结时可接续未完成配置。确认阶段遇到中断可接续，但完成后拒绝重跑；若协议变化，必须启用新的研究与未查看的确认数据，不能删保护文件继续优化本确认集。

第二轮保留 autoresearch 的可验证迭代、固定评价、验证选择、独立确认与完整日志；`research-results.tsv` 记录各方法是否改善验证均值。因为这是预先声明的消融研究，即便均值未改善也保留对应模型供比较。第一轮使用的逐提交保留/撤销循环保持不变。

`research/` 中：`protocol.json` 固定问题和统计计划；`data.py` 构建新划分；`patterns.py` 计算精确二项规则与跨订单规则；`features.py` 生成规则与聚类特征；`models.py` 定义信息组消融；`experiment.py` 隔离验证与确认；`analysis.py` 负责预定统计；`auditing.py` 从原始 CSV 独立重建候选与标签；`reporting.py` 生成表图和证据稿。运行目录为 `autoresearch/research-v2/`。

`research/exploratory.py` 是看到确认集覆盖内收益后增加的事后诊断：只分解已经冻结排序的命中变化，不拟合、不重选。与事先声明的主要比较分开标注，时间与动机保存在 `reports/exploratory_provenance.json`。本机依赖版本见 `requirements-study-lock.txt`。22 项协议测试以及 `scripts/check_demo.py` 的正常/无复购目标演示检查均已通过。

已确认的核心证据：近期衰减频次比累计频次提高 3.73 个百分点；本次购买节奏特征没有额外增益，全局商品统计有约 0.50 个百分点的小幅增益；规则的局部收益未转为总体收益；在规模重叠人群中统一候选数与目标数的分箱权重后，历史长度组的原始表现排序反转。它们是预测与评价层面的发现，不是购买行为的因果结论。

演示自动优先展示第二轮独立确认结果，购物清单显示验证选定方法的种子 42 单个模型。研究标签页解释统计区间、历史长度混杂和复购/整篮指标的区别。

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

耗时字段只测量缓存特征上的批量打分，不包含特征生成、文件读取或排序，不能解释为完整在线响应延迟。

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

## 已运行的第一轮结果

运行目录：`autoresearch/loop-261109-initial/`。1 个初始基线、7 次实验，3 次保留、4 次放弃，无 fork。验证集 Recall@10 从 0.518693 提升至 0.574864，绝对提升 0.056171，相对提升约 10.83%。验证集选定 `lightgbm_base`，之后冻结选择并一次性评价八组方法。

| 方法 | 验证 Recall@10 | 独立测试 Recall@10 |
|---|---:|---:|
| 个人频次 | 0.518693 | 0.506114 |
| 上一单优先 | 0.516657 | 0.520631 |
| 衰减频次 | 0.548947 | 0.551136 |
| 逻辑回归 | 0.563570 | 0.567398 |
| LightGBM 基础 | **0.574864** | **0.576696** |
| LightGBM + 聚类 | 0.571356 | 0.575618 |
| LightGBM + 规则 | 0.572079 | 0.575927 |
| LightGBM + 两者 | 0.572014 | 0.574111 |

测试集 1,500 用户，其中 102 人的下一单没有复购商品，因此 Recall/NDCG 的有效用户数为 1,398。基础 LightGBM 相对个人频次的 Recall@10 差值为 +7.06 个百分点，配对 bootstrap 95% 区间约为 [+5.90, +8.30] 个百分点。

三个附加特征消融相对基础 LightGBM 的差值区间均包含 0。正确表述是“在本次样本和参数下没有观察到稳定增益”，不能由此宣称分群/规则永远无效，也不能把小幅均值下降写成显著退化。

训练 K-means 的轮廓系数约 0.0497；1,224 条规则覆盖约 10.14% 的候选。它们提示群体分离较弱和规则覆盖有限，但尚不能证明这就是没有增益的原因。规则结果和聚类画像均已保存。

本轮仅一个用户抽样种子、固定 K 和规则参数。下一阶段若根据已见测试结果提出新假设，应预留新的独立确认样本，当前测试集转为已见研究数据，避免反复调参后继续声称独立测试。

独立核验已保存结果（不重新训练、不重选模型）：

```powershell
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
```

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
