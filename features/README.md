# 特征与数据合同

本目录保存可复现特征构建代码、合同和必要的派生产物。原始主办方数据不在这里；公共数据库原始响应位于 `../../local-artifacts/references/external-knowledge/raw/`。

## 模块

| 模块 | 用途 | 最终推理关系 |
|---|---|---|
| `compound-structure/` | PubChem 身份、RDKit 描述符和 Morgan 指纹 | 最终 YeaFiLM-CRVAE 使用 |
| `b16-e7-oof-residual/` | E7 OOF 预测及 fold 训练产物 | 训练/审计使用，最终 test 推理不读取 OOF 预测 |
| `compound-knowledge/` | ChEMBL 机制、靶点和活性候选 | 研究候选，不属于最终运行依赖 |
| `strain-genome/` | 菌株群体基因组与蛋白组候选特征 | 研究候选，不属于最终运行依赖 |
| `protein-identity/` | 序列、GO 和物理 PPI 候选特征 | 研究候选，不属于最终运行依赖 |
| `dose-audit/` | 剂量字段与来源审计 | 数据边界审计 |
| `external-knowledge-audit/` | 外部模块使用政策与边界汇总 | 审计 |

`b16-e7-oof-residual/e7-oof-predictions.npz`、`folds/` 和日志是可再生成的大体积本地训练产物，已被 Git 忽略。不要将它们误认为提交时必须上传的模型权重。

机器可读模块清单、原始来源位置和数据边界见 `catalog.json`。刷新命令：

```bash
python -B solution/methods/_shared/workspace_catalog.py
```
