# 常见问题排查

## `CUDA requested but CUDA is unavailable`

改用 `--device cpu`，或确认主办方的 CUDA/PyTorch 构建匹配：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

## `E7 and CVAE protein axes differ`

两个 checkpoint 不是同一训练合同，不能组合。恢复原始成对的 `models/e7/model.pt` 与 `models/crvae/model.pt`，不要手工改蛋白列。

## `CVAE checkpoint was not trained against this E7 checkpoint`

CR-VAE 绑定了特定 E7 SHA-256。必须使用对应 E7 权重，或先按 `docs/RETRAINING.md` 重新训练 CR-VAE。

## `Structure artifact compound axis differs from contract.csv`

NPZ 和 CSV 的化合物顺序、CID 或数量被改变。恢复包内原始的 `fingerprints.npz` 与 `contract.csv`，不要用 pandas 排序其中一个文件。

## `Metadata is missing ...` 或 `pert_time must be positive`

检查列名、编码和数值格式。类别值必须保留官方原始拼写；`pert_time` 必须为正数，不能预先 log2，因为程序会自动执行 log2 和 checkpoint 固定标准化。

## `sha256sum -c audit/SHA256SUMS` 失败

说明包内容被改写、截断或与说明版本不一致。不要继续评测；重新解压正式 ZIP，确认磁盘空间和传输校验后再运行。

## W&B 报账号或网络错误

使用脚本默认的 offline 模式，不需要网络：

```bash
--wandb-mode offline
```

不要把 token 写进命令、配置、日志或提交包。

## 输出行数或列数不对

确认输入 metadata 没有被重复读入、没有额外索引列，且输出使用的是同一个 checkpoint。标准正式测试输出应为 `sample_count` 行、4,423 列（`sample_ID` 加 4,422 个蛋白）。

## 能否把 GO/STRING/ChEMBL 或 OOF 文件放进推理目录？

不需要。最终 `predict.py` 不读取这些材料；加入它们不会提高可复现性，反而可能混淆最终依赖边界。
