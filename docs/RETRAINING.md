# 从官方 train/validation 数据重新训练

本文件只描述训练和验证复现，不改变最终提交的两个正式 checkpoint。建议把重新训练结果写入新的目录，例如 `retraining/20260818-e7-crvae/`，不要覆盖 `models/e7/model.pt` 或 `models/crvae/model.pt`。

## 1. 训练数据和边界

包内训练输入为：

```text
data/official/WAYB_WAYC_metadata_train_val(1).csv
data/official/WAYB_WAYC_proteome_raw_train_val.csv
data/compound-structure/fingerprints.npz
data/compound-structure/contract.csv
data/compound-structure/manifest.json
```

`metadata_train_val(1).csv` 有 8,958 行、15 列；`proteome_raw_train_val.csv` 有同样的 8,958 行和 5,243 个蛋白列加 `sample_ID`。训练程序通过 `sample_ID` 对齐两个文件，并依据 `split_final` 使用官方 train/validation 划分。训练期的类别、`pert_time` 的 log2 均值/尺度、结构描述符尺度和基础蛋白 profile 只能从 `split_final=train` 的非 QC 行估计。

包中没有 test proteome 真值。重训练命令不接受 test proteome 参数，也不会从 test metadata 估计统计量。

## 2. E7/YeaFiLM 阶段

```bash
bash scripts/train_e7.sh retraining/e7-run cuda:0
```

如果 `python` 不是安装 PyTorch 的解释器，可写成 `PYTHON_BIN=/opt/conda/envs/goai/bin/python bash scripts/train_e7.sh retraining/e7-run cuda:0`；`train_crvae.sh`、`retrain_all.sh`、`smoke_crvae_training.sh` 和 `verify_bundle.sh` 同样支持该变量。

脚本调用 `code/methods/b10-a2-film-cross-mlp/run.py`，传入包内 train/validation metadata、proteome、冻结 RDKit 特征、`config.json` 和官方六模块 scorer。主要固定配置包括 seed=42、hidden/latent/fusion 维度、dropout、最大 epoch 和 early stopping。程序会写出 checkpoint、训练历史、验证预测、验证评分、配置快照和 manifest。

`manifest.json` 中的 `retained_proteins`、输入哈希和 feature manifest 是 CR-VAE 阶段的接口，不要手工编辑。

## 3. CR-VAE 阶段

先完成 E7，再把 E7 输出目录传给：

```bash
bash scripts/train_crvae.sh retraining/crvae-run retraining/e7-run cuda:0
```

CR-VAE 会加载 `retraining/e7-run/model.pt`，冻结 E7，只训练 Conditional Residual VAE 的新增参数。它会检查 E7 蛋白轴、E7 manifest 和 E7 core；如果 E7 checkpoint 与 manifest 不匹配会停止。训练目标使用训练集 paired-control delta 和训练集真值，验证阶段只使用官方 validation 行；最终推理调用 conditional-prior mean，不调用 posterior。

输出包括：

```text
retraining/crvae-run/model.pt
retraining/crvae-run/manifest.json
retraining/crvae-run/config.snapshot.json
retraining/crvae-run/training-history.json
retraining/crvae-run/latent-audit.json
retraining/crvae-run/validation-prediction.csv
retraining/crvae-run/six-module.json
retraining/crvae-run/result.json
```

## 4. 一键两阶段训练

```bash
bash scripts/retrain_all.sh retraining/full-run cuda:0
```

该命令依次创建 `retraining/full-run/e7/` 和 `retraining/full-run/crvae/`，并将第一阶段的 checkpoint 自动绑定到第二阶段。CPU 也可运行，但 4,422 个蛋白输出和全量训练明显较慢；可在不改模型定义的前提下使用兼容 PyTorch 版本。

## 5. 训练 smoke test

只检查真实数据加载、结构轴、E7 权重、paired-control 构造和一批 CR-VAE 梯度，不进入完整优化：

```bash
bash scripts/smoke_crvae_training.sh verification/crvae-smoke cuda:0
cat verification/crvae-smoke/smoke-training.json
```

应看到 `gradients_finite=true`、`additional_parameter_budget_pass=true`、`validation_posterior_calls=0`。这不是最终模型训练，也不替代完整 `train_crvae.sh`。

## 6. W&B 和可重复性

脚本默认 `--wandb-mode offline`，因此不需要账号、token 或网络。若主办方允许在线记录，可显式改为 `--wandb-mode online`，但必须自行配置凭据，不要将凭据写入包或日志。每次训练应保存命令、Python/PyTorch 版本、设备、配置、输入哈希、输出哈希、随机种子和 scorer 配置。

## 7. 重训练结果与正式模型的关系

重新训练得到的权重不应自动替换正式权重。只有在重新计算官方允许的 validation 评测、核对模型/特征/数据合同并人工确认后，才可以把新的 checkpoint 作为候选版本。最终无真值推理默认继续使用 `models/e7/model.pt` 和 `models/crvae/model.pt`。
