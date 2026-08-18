# 包含与排除清单

## 包含

- 最终 YeaFiLM 和 CR-VAE 训练/推理源码、配置与测试
- E7 seed42 checkpoint 及其 manifest、结果、训练历史和六模块指标
- CR-VAE seed42 checkpoint 及其 manifest、结果、训练历史、潜变量审计和六模块指标
- 官方 train/validation metadata 与蛋白质组矩阵，用于私有训练复现
- 官方 test metadata，用于无真值推理
- PubChem 派生的冻结属性、名称/CID 合同、RDKit 特征合同、Morgan 指纹和描述符；原始 PUG-REST 响应不在本包内
- `FINAL_DEPENDENCIES.json`：最终推理必需、私有复现、训练期和研究期依赖的角色与 SHA-256 清单
- 部署、输入输出合同、重训练、故障排查、运行入口、架构图、技术说明和文件校验值
- `train_e7.sh`、`train_crvae.sh`、`retrain_all.sh` 和真实数据 smoke test，供主办方从 train/validation 数据重建 checkpoint

## 明确排除

- `WAYB_WAYC_proteome_raw_test.csv` 及任何 test 蛋白质组真值
- API token、Cookie、密码、W&B 凭据和其他秘密
- W&B 在线/离线运行目录
- validation prediction 大文件、无关候选模型和无关实验缓存
- Git 元数据及公开仓库 remote

## 发布规则

本目录是私有运行审计包。官方数据和 checkpoint 不得复制进公开源码仓库；对外发布前必须重新按提交平台字段、主办方数据许可和文件大小限制生成专用包。
