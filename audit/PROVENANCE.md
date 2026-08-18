# 来源与版本记录

## 主办方文件

以下文件来自 GOAI 2026 虚拟酵母扰动任务的主办方数据包，仅保存在私有运行审计包中：

| 文件 | SHA-256 | 用途 |
|---|---|---|
| `WAYB_WAYC_metadata_train_val(1).csv` | `9414f22d71e925a3b85544b49fde252613c87808d34738a84785003adb8131ef` | train/validation 条件、冻结划分与训练合同 |
| `WAYB_WAYC_proteome_raw_train_val.csv` | `a15d9a40a6ad4e8e84a4ce4ed08644fce78780d31ace5561928517c4a5fa7ccb` | train 标签与允许范围内的 validation 早停/评估 |
| `WAYB_WAYC_metadata_test(1).csv` | `42f2df9ea79f28da8344e96b5181edacc215744a858d1a4eaa729c2e1cc69d31` | 无真值推理输入 |

未包含 `WAYB_WAYC_proteome_raw_test.csv`。

## 外部公开资源

- 数据库：PubChem
- 接口：PUG-REST
- 获取日期：2026-08-16
- 实体数：56 个化合物
- 原始响应 SHA-256：`c1185c82564a1c699ca8941e5a47849834cfb6e4a351d8429cae02e1d5ccc24b`
- RDKit：2026.03.5
- Morgan：radius 2，2048 bit，包含手性
- 描述符：分子量、LogP、TPSA、HBD、HBA

完整化学身份和标准化约定见 `code/EXTERNAL_DATA.md`。

## 模型权重

| 模型 | 实验 | SHA-256 |
|---|---|---|
| E7/YeaFiLM | `20260816-1658_b10-a2-film-cross-mlp-seed42` | `7f567e9b7d959459e56123ed464eaa101498ae52654bec42b4e981d49da0faf0` |
| Conditional Residual VAE | `20260816-2017_b16-a2-conditional-residual-vae-seed42` | `501e1158332b464c86daddb56b9a1837be19dfcfc2ddd315a5943734d205a9ad` |

CR-VAE 依赖冻结 E7 checkpoint，因此两个权重都必须保留才能直接运行最终推理。

## 生成预测

- 生成日期：2026-08-17
- 形状：4,454 样本 × 4,423 列（`sample_ID` + 4,422 蛋白）
- SHA-256：`6792632df582d2a4ae707a0c18021e935189de9574719e47cd38ad4b338c6ef5`
- 审计：`truth_loaded=false`，`posterior_called=false`
