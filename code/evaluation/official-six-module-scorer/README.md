# GOAI 2026 六模块最终本地评分器

状态：`official-faithful provisional scorer 1.1.0`。

它严格实现 2026-08-14 当前手册已经明确的内容：

- 六模块及官方权重：20% / 25% / 20% / 20% / 10% / 5%；
- 逐样本绝对保真度 PCC/R²；
- 相对严格匹配真实对照的 log2 fold change PCC；
- 仅由 `split_final=train` 拟合的上下文均值与药物均值参考；
- `val_both` 和 `val_time` 的 FC + 绝对保真；
- `|Δ_true| > 1` 的方向、PCC、Recall@K、F1 和 AUPRC；
- validation truth 只在预测冻结后评分，不进入 reference 拟合；
- 默认拒绝独立 evaluation/test truth 文件。

`1.1.0` 进一步冻结当前官方 train/validation 输入的 SHA-256、split 计数和 5,243 蛋白
feature contract，并拒绝未知 split、未知样本、未知蛋白、重复列和 raw 中的正负无穷。

由于主办方尚未公开可执行评分代码、模块内部权重、多个 control 的合并方式和 Water/DMSO 映射，输出总分字段刻意命名为：

```text
provisional_weighted_proxy
provisional_weighted_proxy_100
```

这是一版可用于模型选择的统一本地总分，不声称逐位复现组委会隐藏分数。官方一旦发布精确 scorer，只需替换配置/聚合层。

## 核心公式

### M1 绝对保真度

本 profile 对每个 validation treatment 样本跨保留蛋白计算：

\[
r_i=\operatorname{PCC}(y_i,\hat y_i),\qquad
R_i^2=1-\frac{\sum_j(y_{ij}-\hat y_{ij})^2}{\sum_j(y_{ij}-\bar y_i)^2}.
\]

默认模块值为 `mean(mean_i(r_i), mean_i(R²_i))`。原始负值保留；仅在代理总分中截到 `[0,1]`。
手册写的是“全部划分”，但没有明确 control 是否进入绝对保真度，因此 treatment-only 是本地
版本化范围假设，不伪装成已公开官方细节。

### M2 匹配对照原始 FC

在 log2 空间：

\[
Δ_{true}=y_{treat}-y_{control},\qquad
Δ_{pred}=\hat y_{treat}-y_{control}.
\]

对每个样本计算 `PCC(Δ_true, Δ_pred)` 后聚合。对照按来源、菌株、培养基、温度、时间、仪器和板号严格匹配；时间单位必须一致。

### M3 上下文均值残差

只由 train 中匹配到 train control 的 treatment delta 计算：

\[
μ_{ctx}=\operatorname{mean}_{train\ drugs\ in\ same\ context}(Δ_{true}).
\]

在 `val_chem_only` 计算：

\[
\operatorname{PCC}(Δ_{pred}-μ_{ctx},Δ_{true}-μ_{ctx}).
\]

当前 profile 对匹配到的 train treatment 样本做算术均值，因此重复更多的药物样本权重更高；
“先按唯一药物等权再求均值”并未被手册机器化规定，已作为公开假设记录。

### M4 药物均值残差

只由 train 计算：

\[
μ_{drug}=\operatorname{mean}_{train\ contexts\ for\ same\ drug}(Δ_{true}).
\]

在 `val_strain_only` 计算对应残差 PCC。

这里同样是 matched train treatment sample 加权，而不是先对唯一上下文等权。

### M5 双重未知与时间插值

分别在 `val_both` 与 `val_time` 计算 M1 式绝对保真和 M2 式 FC PCC。当前主 profile 将
`both 绝对保真`、`both FC`、`time 绝对保真`、`time FC` 四个复合量等权；每个绝对保真
再由 PCC/R² 等权组成，因此机器配置中展开为 6 个 component。这是版本化本地假设。

### M6 高效应蛋白与 DEP

正类严格定义为：

\[
|Δ_{true}|>1,
\]

不是 `>=1`。输出方向准确率、高效应 PCC、precision、recall、F1、average precision、梯形 PR-AUC、Recall@真实 DEP 数量及 Recall@10/50/100。模块代理值对方向准确率、高效应逐样本 PCC、F1、average precision 等权；Recall 单独报告但不单独决定模块分。

## 运行

```bash
evaluation/official-six-module-scorer/scripts/score_validation.sh \
  competition-materials/raw-data/virtual-yeast-perturbation-proteomics/extracted/input/WAYB_WAYC_metadata_train_val\(1\).csv \
  competition-materials/raw-data/virtual-yeast-perturbation-proteomics/extracted/input/WAYB_WAYC_proteome_raw_train_val.csv \
  /path/to/frozen-validation-prediction.csv \
  evaluation/official-six-module-scorer/configs/current-handbook-sample-mean-v1.json \
  /path/to/six-module-result.json
```

预测文件要求：

- 第一列/索引列为 `sample_ID`；
- 至少覆盖全部 validation treatment 样本；
- 至少包含由 train-only `<80%` 缺失率规则保留的蛋白列；
- 数值必须全部 finite，且明确为 log2 尺度；
- 可以包含被 train-only 过滤掉的其他官方蛋白列，评分器只选择保留列表；任何不在官方 raw
  feature contract 中的未知蛋白列都会拒绝；
- 当前主配置还会核对官方 train/validation metadata、proteome 的 SHA-256、split 计数和
  原始 5,243 蛋白列数。若主办方更新数据，必须新增或升级 profile，不能静默沿用旧合同。

注意：仅凭一组数值无法数学上唯一证明其一定来自 log2 变换；评分器能做的是冻结输入来源、
要求 prediction provenance 明确声明 log2，并报告预测值范围。raw intensity 或 z-score 不得传入。

## 输出

主 JSON 包括：

- 0–1 和 0–100 代理总分；
- M1–M6 的 raw score、normalized score、weighted points；
- 各 component 的有限值 raw aggregate、未定义样本补 0 后的 proxy value，以及归一化值；
- 每个模块全部内部指标；
- 每个验证场景的绝对指标和 FC 指标；
- control、context reference、drug reference 覆盖率；
- 输入、样本轴、蛋白轴、配置和 train reference 哈希；
- 所有本地聚合假设和官方未决项。

默认还会生成：

```text
<result>.details/sample_metrics.csv.gz
<result>.details/control_matches.csv.gz
<result>.details/train_reference_manifest.json
<result>.details/score_summary.csv
<result>.details/component_metrics.csv
<result>.json.sha256
```

`score_summary.csv` 第一行是 0–100 总分，随后是 M1–M6 的 raw score、normalized score 和加权得分；`component_metrics.csv` 展开每个模块内部的 PCC、R²、FC、残差和 DEP 组件。

## 数值纪律

- CSV 读取与所有公式均使用 float64，不先降精度到 float32；
- 所有比较使用 pairwise-finite mask；
- 常量向量 PCC/R²返回 undefined，在 JSON 中写 `null`；
- raw 指标以有限样本聚合并保留未定义数量；score-bearing proxy 将未定义样本补 0，且不重新
  分配样本或模块权重，防止通过常量预测跳过困难样本；
- 原始负 PCC/R²保留；代理总分中的相关/R²组件截到 `[0,1]`；
- PertPy 只作为可选分布诊断，不调用其名为 `root_mean_squared_error` 的欧氏距离别名。

## 测试

```bash
cd evaluation/official-six-module-scorer
PYTHONPATH=src pytest -q
```
