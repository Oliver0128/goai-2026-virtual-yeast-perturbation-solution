# 旧材料为何没有最终 CR-VAE checkpoint

这不是 CR-VAE 训练失败或权重丢失。

1. 旧目录 `20260816-2151-e7-preliminary-materials` 在 2026-08-16 21:51 创建，角色明确是 E7 初赛材料快照；其 README、readiness 审计和目录内容均以 E7 为中心。
2. 最终方案后来选定为冻结 E7 主干加 B16-A2 Conditional Residual VAE。正式 CR-VAE checkpoint 实际已于 2026-08-16 20:17 生成，但旧 E7 快照没有在模型选择和文档改写后重新组装。
3. 2026-08-16 23:35 整理独立 solution 仓库时，目标是形成可公开的源码仓库。按数据和 Git 边界，CSV、NPZ、PT/PTH/CKPT、官方数据与测试真值均被 `.gitignore` 明确排除。
4. 因而形成了两个互补但未合并的产物：旧私有包含 E7 数据/权重但不含最终 CR-VAE；公开源码目录含最终 CR-VAE 源码和无真值入口但不含任何数据或权重。

本目录将二者按私有复现审核边界重新合并，并加入最终 CR-VAE checkpoint、冻结结构特征、完整 test metadata 推理结果及校验清单。
