# 验证结果

验证日期：2026-08-18（交付前复核）

## 最新交付前复核

- 新增并通过 shell 语法检查：`train_e7.sh`、`train_crvae.sh`、`retrain_all.sh`、`smoke_crvae_training.sh`、`verify_bundle.sh`；脚本均支持 `PYTHON_BIN` 显式选择 PyTorch 环境。
- 真实数据 CR-VAE smoke test：8 行训练样本，context `(8, 256)`，latent `(8, 64)`，protein `(8, 4422)`，`gradients_finite=true`，新增参数 `1,148,070`，`validation_posterior_calls=0`。
- 原私有审计包的完整两阶段重训练门禁已通过：E7 127 epochs、CR-VAE 41 epochs，均从包内 train/validation 数据启动并产出 `model.pt`、manifest、训练历史、验证预测和六模块评分；本复制目录按要求不含这些官方原始数据，重训练时需由用户按官方渠道补齐；新 CR-VAE 的 `frozen_e7_model_sha256` 与新 E7 checkpoint 一致。
- 最新无真值推理和原有提交测试均重新执行；未读取 test proteome 真值。
- 独立 ZIP 解压验收：从全新临时目录运行 `scripts/verify_bundle.sh quick`，全部 checksum 和测试通过；随后运行 `scripts/predict_test.sh`，输出 `(4454, 4423)`，输出 SHA-256 为 `6792632df582d2a4ae707a0c18021e935189de9574719e47cd38ad4b338c6ef5`。

## 源码门禁

- YeaFiLM 测试：5 passed，1 skipped
- CR-VAE 测试：7 passed
- submission 测试：2 passed
- 六模块评分器测试：76 passed
- Shell 语法检查：通过（5 个运行脚本）
- 包内全部 SHA-256：通过

5 个 skipped 测试是仓库测试针对可选冻结数据资产设置的条件跳过；本私有包已包含实际结构特征，并通过全量推理验证。

## 全量无真值推理

- 输入 test metadata 样本：4,454
- 输出列：4,423（`sample_ID` + 4,422 蛋白）
- 重复 sample ID：0
- 非有限预测值：0
- `truth_loaded=false`
- `posterior_called=false`
- 输出 SHA-256：`6792632df582d2a4ae707a0c18021e935189de9574719e47cd38ad4b338c6ef5`

## 权重与结构合同

- E7 checkpoint 与正式实验 manifest 记录一致
- CR-VAE checkpoint 与正式实验 result/artifact-hashes 记录一致
- fingerprints.npz 与冻结外部资源记录一致
- contract.csv 与冻结外部资源记录一致

## 边界

- 未包含 test 蛋白质组真值
- 未包含凭据或 W&B 运行目录
- 未修改公开 solution 仓库
- 未执行 Git add、commit 或 push
