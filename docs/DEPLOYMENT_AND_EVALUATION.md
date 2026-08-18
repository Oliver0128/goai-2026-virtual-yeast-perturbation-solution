# YeaFiLM-CRVAE 部署、复现与评测说明

版本：2026-08-18
适用包：`20260817-1531-final-crvae-runtime-audit`

训练复现请配合阅读 [`RETRAINING.md`](RETRAINING.md)；字段级输入/输出合同见 [`INPUT_OUTPUT_CONTRACT.md`](INPUT_OUTPUT_CONTRACT.md)；异常排查见 [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)。

本文是给主办方复现和评测使用的操作说明。所有路径均相对于本说明所在的运行包根目录（即包含 `code/`、`data/`、`models/` 的目录）。最终推理是无真值的两阶段冻结流水线：先运行 E7/YeaFiLM，再将其残差和融合表示交给 CR-VAE 的 conditional-prior mean 解码器。

## 1. 运行包结构

```text
code/submission/predict.py                         推理入口
code/methods/b10-a2-film-cross-mlp/core.py        E7 模型定义
code/methods/b16-a2-conditional-residual-vae/core.py  CR-VAE 定义
code/requirements.txt                              参考 Python 环境
data/inference-metadata/WAYB_WAYC_metadata_test(1).csv  测试元数据（不含测试真值）
data/compound-structure/fingerprints.npz           56 个化合物的冻结结构特征
data/compound-structure/contract.csv               结构特征轴和 PubChem/RDKit 合同
models/e7/model.pt                                 E7 checkpoint
models/crvae/model.pt                              CR-VAE checkpoint
scripts/predict_test.sh                             一键测试推理
audit/FINAL_DEPENDENCIES.json                       依赖角色、边界和哈希总表
audit/SHA256SUMS                                   包内逐文件哈希
```

两个 `.pt` 文件不能互换或合并。CR-VAE checkpoint 内含冻结的 E7 checkpoint 和 E7 core SHA-256 约束，推理启动时会主动检查这两个绑定。

## 2. 环境部署

### 2.1 推荐参考环境

运行包记录的参考环境为 Python 3.13.14，主要版本见 `code/requirements.txt`：NumPy 2.4.6、pandas 3.0.3、PyTorch 2.13.0+cu132、RDKit 2026.3.5。PyTorch CUDA wheel 是否可直接取得取决于主办方的软件源；若使用 CPU，可安装与 Python 3.13 兼容的 CPU 版 PyTorch，模型接口不变。

```bash
cd /path/to/goai-2026-virtual-yeast-perturbation-solution
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
```

如果主办方有多个 Python 环境，所有启动脚本都支持显式指定解释器，例如 `PYTHON_BIN=/opt/conda/envs/goai/bin/python bash scripts/predict_test.sh`。不指定时使用当前 PATH 中的 `python`。

如果平台不能提供 `torch==2.13.0+cu132`，请使用平台批准的 PyTorch 2.x CPU/CUDA 构建，并记录完整的 `python -m pip freeze`；不要更改 checkpoint、特征维度或模型配置。推理本身不重新访问 PubChem，也不需要联网。

### 2.2 环境自检

```bash
python - <<'PY'
import numpy, pandas, torch
print('numpy', numpy.__version__)
print('pandas', pandas.__version__)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
PY
```

若要从 PubChem 结构重新生成特征，还需 RDKit；正式推理直接使用包内冻结的 `fingerprints.npz`，不依赖网络或在线数据库。

## 3. 完整性检查和权重放置

解压后先在运行包根目录执行：

```bash
sha256sum -c audit/SHA256SUMS
```

应全部显示 `OK`。权重必须保持以下相对路径和文件名：

```text
models/e7/model.pt       # E7/YeaFiLM，SHA256 见 audit/FINAL_DEPENDENCIES.json
models/crvae/model.pt    # Conditional Residual VAE，SHA256 见同一文件
```

如果主办方将权重放在其他位置，可以通过命令行显式传入 `--e7-model` 和 `--cvae-model`，但不得替换其中任一 checkpoint，也不得只加载其中一个。`predict.py` 会检查：

- E7 和 CR-VAE 的 4,422 个蛋白质轴完全一致；
- CR-VAE 记录的 E7 权重哈希等于实际 E7 文件哈希；
- CR-VAE 记录的 E7 core 源码哈希等于实际源码。

## 4. 输入数据规范

### 4.1 元数据 CSV

`--metadata` 必须是 UTF-8 CSV，至少包含以下列：

| 列 | 类型/约束 | 用途 |
|---|---|---|
| `sample_ID` | 字符串，整列唯一 | 输出主键；不能重复 |
| `Strains` | 字符串 | 菌株 one-hot；未知类别按全零块处理 |
| `perturbation_no_concentration` | 字符串 | 化合物身份；必须与 `contract.csv`/checkpoint 词表一致或可标记为未知 |
| `Medium` | 字符串 | 培养基类别 |
| `Temperature` | 数值/字符串 | 温度类别 |
| `pert_time` | 正数 | 先做 `log2`，再使用 checkpoint 固定均值和尺度标准化 |
| `pert_time_unit` | 字符串 | 时间单位类别 |
| `data_source` | 字符串 | 数据来源类别 |
| `instrument` | 字符串 | 仪器类别 |

官方完整 metadata 通常还包含 `data_source`、`pert_id`、`Yeast_cell_plate`、`protein_well`、`split_final`、`strain_role`、`chemical_role` 等列。推理代码允许这些额外列存在，但普通推理不会使用它们；只有使用 `--treatment-only` 时才需要 `pert_id`，并会排除 Water、DMSO、Quality Control 和 `pert_id=48`。

列名、类别拼写和化合物名称必须保持原样。缺少必需列、`sample_ID` 重复、`pert_time<=0`、数值非有限或 checkpoint 维度不匹配时，程序会直接报错，不会静默补值。

示例（仅展示格式，不代表完整数据）：

```csv
sample_ID,Strains,Medium,Temperature,pert_time,pert_time_unit,perturbation_no_concentration,data_source,instrument
B1,CRD,YNB+CSM+2% galactose,30,15,min,Water,WAYB,O
B2,CRD,YNB+CSM+2% galactose,30,15,min,Amphotericin B,WAYB,O
```

### 4.2 结构特征输入

`--structure-npz` 和 `--structure-contract` 必须成对使用：

- `fingerprints.npz` 必须包含 `competition_names`、`pubchem_cids`、`morgan_bits`、`descriptor_names`、`descriptors_raw` 五个数组；
- `morgan_bits` 形状为 `(56, 2048)`、类型为 `uint8`、值只能是 0/1；
- `descriptors_raw` 形状为 `(56, 5)`，五列为分子量、LogP、TPSA、HBD、HBA；
- `contract.csv` 的化合物顺序和 PubChem CID 顺序必须与 NPZ 完全一致；
- 结构来源和 RDKit 处理规则见 `data/compound-structure/manifest.json`。

正式推理不要求重新运行 RDKit，也不要求重新请求 PubChem。若主办方自行重建该文件，必须复现同一 PubChem/RDKit 版本、化合物别名解析、FragmentParent 例外、Morgan 参数和文件轴顺序。

## 5. 推理命令

### 5.1 推荐一键运行

```bash
bash scripts/predict_test.sh
```

该脚本使用包内默认路径、`--device auto` 和 batch size 512，生成：

```text
outputs/test-prediction.csv
outputs/inference-audit.json
```

### 5.2 显式命令

```bash
python code/submission/predict.py \
  --metadata data/inference-metadata/WAYB_WAYC_metadata_test\(1\).csv \
  --structure-npz data/compound-structure/fingerprints.npz \
  --structure-contract data/compound-structure/contract.csv \
  --e7-model models/e7/model.pt \
  --cvae-model models/crvae/model.pt \
  --output outputs/test-prediction.csv \
  --audit outputs/inference-audit.json \
  --device auto \
  --batch-size 512
```

可选参数：

- `--device cpu`：强制 CPU；`--device cuda`：强制 CUDA；`auto`：有 CUDA 就用 CUDA，否则 CPU。
- `--batch-size N`：显存不足时降低 N，不改变结果定义。
- `--treatment-only`：仅在官方提交合同明确要求只输出处理样本时使用；不要自行添加。

## 6. 输出格式与审计

`outputs/test-prediction.csv` 为 UTF-8 CSV：

- 第一列为 `sample_ID`，顺序与输入 metadata 保持一致；
- 后续 4,422 列为 checkpoint 中 `retained_proteins` 的蛋白质名称，顺序不可重排；
- 每个单元格为有限浮点数；不包含额外索引列。

`outputs/inference-audit.json` 会记录输入、结构文件、两个权重和输出的 SHA-256，以及样本数、蛋白质数、设备模式和推理模式。正常结果应满足：

```json
{
  "inference_mode": "frozen E7 plus deterministic CVAE conditional-prior mean",
  "truth_loaded": false,
  "posterior_called": false,
  "protein_count": 4422
}
```

评测时应使用主办方自己的真值和官方 scorer，在外部目录计算指标；不要把真值放入运行包，也不要让推理程序读取任何 test proteome 文件。

## 7. 复现检查清单

```bash
# 1. 文件完整性
sha256sum -c audit/SHA256SUMS

# 2. 单元测试
python -m pytest -q code/submission/test_predict.py

# 3. 无真值推理
bash scripts/predict_test.sh

# 4. 输出基本检查
python - <<'PY'
import json, pandas as pd
pred = pd.read_csv('outputs/test-prediction.csv')
audit = json.load(open('outputs/inference-audit.json'))
assert pred['sample_ID'].is_unique
assert pred.shape[1] == 4423
assert pred.shape[0] == audit['sample_count']
assert audit['protein_count'] == 4422
assert audit['truth_loaded'] is False
assert audit['posterior_called'] is False
print(pred.shape, 'audit OK')
PY
```

若使用 GPU，建议同时记录 `nvidia-smi`、Python/PyTorch 版本、命令行、输入哈希和输出哈希。CPU/GPU 只影响执行设备，不改变输入合同和输出列合同。

## 8. 评测边界与不应执行的操作

- 不需要 GO/QuickGO、STRING/PPI、ChEMBL、菌株基因组、蛋白身份注释、OOF 预测或 fold 权重才能完成最终推理。
- 不需要联网下载 PubChem；当前包中的结构文件是冻结派生数据。
- 不要用 train/validation/test 标签重新拟合标准化参数、筛选蛋白或改变化合物词表。
- 不要把 `WAYB_WAYC_proteome_raw_train_val.csv` 当作 test 输入；它仅用于私有复现审计。
- 不要删除 `audit/`，否则无法核验来源、哈希和无真值边界。

最终依赖角色、checkpoint 哈希、数据来源和 Git/分发边界，以 `audit/FINAL_DEPENDENCIES.json` 和 `audit/SHA256SUMS` 为准。
