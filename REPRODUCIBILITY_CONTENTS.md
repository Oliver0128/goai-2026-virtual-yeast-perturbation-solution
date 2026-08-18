# 可复现结果包内容与数据边界

本目录由 2026-08-18 最终 YeaFiLM-CRVAE 运行审计包及比赛工作区中的派生特征资产整理而成。

## 已包含

- `models/e7/model.pt`：YeaFiLM 基础模型正式 checkpoint。
- `models/crvae/model.pt`：Conditional Residual VAE 正式 checkpoint。
- 两个模型对应的配置快照、manifest、训练历史、评分结果和审计记录。
- `code/`、`scripts/`、`docs/`：部署、推理、重训练、评分与测试代码。
- `data/compound-structure/`：冻结的 PubChem/RDKit 指纹、描述符、标识映射、contract 和 manifest。
- `data/inference-metadata/`：仅用于无真值推理的 test metadata。
- `features/`：研究期化合物知识、蛋白身份/GO/STRING、蛋白序列、菌株蛋白组序列、菌株基因组派生特征、OOF 特征和相应构建脚本/合同。
- `outputs/test-prediction.csv` 与 `outputs/inference-audit.json`：最终无真值预测和运行审计。

## 明确排除

- 官方 train/validation metadata。
- 官方 train/validation 原始蛋白质组矩阵。
- `WAYB_WAYC_proteome_raw_test.csv` 或任何形式的 test 蛋白质组真值。
- 官方原始 ZIP、解压副本、凭据、W&B 密钥、治理文件和缓存目录。

## 使用方式

直接复现最终无真值推理：

```bash
PYTHON_BIN=/path/to/python bash scripts/predict_test.sh
```

重新训练需要用户自行从主办方正式渠道取得下列官方文件并放入 `data/official/`：

```text
WAYB_WAYC_metadata_train_val(1).csv
WAYB_WAYC_proteome_raw_train_val.csv
```

随后运行 `scripts/retrain_all.sh`。本目录中的 test metadata 不得用于训练统计、特征学习、归一化或超参数选择。

## 注释资产角色

最终正式 YeaFiLM-CRVAE 推理实际使用的是 `data/compound-structure/` 中的 PubChem/RDKit 结构特征。`features/protein-identity/`、`features/strain-genome/` 和 `features/compound-knowledge/` 是完整保留的研究期候选先验与注释资产，不应误述为最终 checkpoint 的强制输入。
