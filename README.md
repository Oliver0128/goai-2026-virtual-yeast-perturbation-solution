# YeaFiLM-CRVAE 可复现结果包（不含官方训练数据与 test 真值）

生成日期：2026-08-18（交付前复核版）

本目录用于 GOAI 2026 虚拟酵母扰动任务的本地复现、无真值推理和组委会复现审核准备。它包含正式权重、最终预测结果和研究期注释/序列资产，但不包含官方 train/validation 原始数据或任何 test 蛋白质组真值；包含的 test metadata 只提供无真值推理所需条件字段。它不是公开源码包，权重及大体积资产不应直接提交到公开 Git。

## 最终模型

- 基础网络：B10-A2 YeaFiLM，正式实验 `20260816-1658_b10-a2-film-cross-mlp-seed42`
- 残差模型：B16-A2 Conditional Residual VAE，正式实验 `20260816-2017_b16-a2-conditional-residual-vae-seed42`
- 部署路径：冻结 E7 基础预测，加 CR-VAE conditional-prior mean 残差；推理不调用训练期 posterior。

## 目录

```text
code/                       训练、推理、特征构建和评分源码
data/inference-metadata/    仅含无真值推理需要的 test metadata
data/compound-structure/    冻结 PubChem/RDKit 结构特征
models/e7/                  E7 checkpoint、manifest 和训练记录
models/crvae/               CR-VAE checkpoint、manifest 和训练记录
outputs/                    本次无真值推理及审计输出
features/                   研究期注释、序列、结构及派生特征资产
audit/                      SHA-256、来源、包含/排除清单和验证记录
scripts/                    相对路径运行入口
docs/                       架构图和技术说明
```

## 直接推理

结论：主办方只要安装依赖，即可使用本目录中的两个 checkpoint 直接完成无真值推理、代码测试和格式检查；不需要重新训练，也不需要 test 蛋白质组真值。只有要重新训练或计算验证集分数时，才需要另外取得官方 train/validation 数据。

在本目录执行：

```bash
bash scripts/predict_test.sh
```

输出：

```text
outputs/test-prediction.csv
outputs/inference-audit.json
```

面向主办方的环境部署、权重放置、输入/输出规范、复现命令和评测边界见 [`docs/DEPLOYMENT_AND_EVALUATION.md`](docs/DEPLOYMENT_AND_EVALUATION.md)。

训练复现、重新生成两个 checkpoint 的命令见 [`docs/RETRAINING.md`](docs/RETRAINING.md)。由于本目录按要求排除了官方训练原始数据，重训练前必须从主办方正式渠道取得相应文件并放入 `data/official/`；输入/输出字段合同见 [`docs/INPUT_OUTPUT_CONTRACT.md`](docs/INPUT_OUTPUT_CONTRACT.md)，本目录实际包含/排除清单见 [`REPRODUCIBILITY_CONTENTS.md`](REPRODUCIBILITY_CONTENTS.md)。

推理输入仅包括 test metadata、冻结结构特征和两个 checkpoint，不读取 test 蛋白质组真值。

## 依赖边界

本方案不使用菌株基因组、GO、STRING、通路注释、外部蛋白质组或转录组、蛋白语言模型嵌入及第三方药物响应标签。唯一外部数据库为 PubChem；RDKit 用于从冻结化合物结构生成 Morgan 指纹和分子描述符。

## 分发边界

- `models/` 及 `features/` 下的大体积注释/序列仅供获准的复现审核，不得进入公开 Git。
- 官方 train/validation metadata、train/validation proteome 和 test proteome 真值均未复制到本目录。
- `data/inference-metadata/` 仅包含 test metadata，不包含任何测试蛋白质组测量值。
- 不包含 `WAYB_WAYC_proteome_raw_test.csv` 或任何测试蛋白质组真值。
- 公开源码包应只发布 `code/`、必要文档和外部资源获取/生成说明。
- 若提交平台只要求源代码，不应擅自上传主办方原始数据；checkpoint 是否作为私有附件上传，以平台字段或组委会通知为准。

## 完整性验证

```bash
sha256sum -c audit/SHA256SUMS
bash scripts/verify_bundle.sh quick
```

最终依赖角色和 RDKit/PubChem 边界见 `audit/FINAL_DEPENDENCIES.json`。该文件明确区分最终无真值推理必需文件、需要主办方补充的训练数据、训练期 OOF/fold 产物和研究候选注释；原始 PubChem PUG-REST 响应不随本运行包分发。
