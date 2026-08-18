# 外部数据与化合物特征说明

本结果包使用主办方提供的实验数据训练蛋白质组预测模型，并额外保留公开的化合物结构信息。本目录不分发主办方 train/validation 原始数据或 test 真值；外部信息只用于描述化合物本身，不包含任何蛋白质组测量值、验证集标签或测试集标签。

这份文件记录化合物名称如何对应到 PubChem 条目、结构数据如何冻结、RDKit 如何把结构转换为模型输入，以及复现时需要注意的几个化学身份问题。Python 依赖已经单独写入 [`requirements.txt`](requirements.txt)，不在这里重复列出。

## 1. 使用了什么外部数据

唯一使用的外部数据库是 **PubChem**。我们通过 PUG-REST 接口查询比赛 metadata 中出现的化合物，保存其 PubChem CID、名称、分子式、SMILES、InChI、InChIKey 和常用理化属性。

| 项目 | 记录值 |
|---|---|
| 数据库 | PubChem |
| 查询接口 | PUG-REST |
| 获取日期 | 2026-08-16 |
| 核对后的化合物数量 | 56 |
| 用于建模的结构字段 | Isomeric SMILES |
| PubChem 发布版本 | 接口未返回独立 release identifier |
| 原始响应 SHA-256 | `c1185c82564a1c699ca8941e5a47849834cfb6e4a351d8429cae02e1d5ccc24b` |

PubChem 的在线记录会持续修订，而本次接口响应没有携带可以直接引用的数据库版本号。因此，本项目没有把“PubChem”或访问日期单独当作充分的版本标识。能够唯一指向本次数据快照的是以下四项：

- 获取日期：2026-08-16；
- 经过人工核对的化合物名称与 CID 对照表；
- 保存下来的 PUG-REST 原始 JSON 响应；
- 原始响应文件的 SHA-256。

复现实验时，应优先使用参赛者本地保存的冻结响应。本代码仓库不再分发该响应、生成后的 CSV 或 NPZ；使用者应依据本文件记录的来源、身份合同和校验值自行准备。若重新访问 PubChem，应比较 CID、Isomeric SMILES、InChIKey 和文件校验值，并记录任何变化。重新查询结果不得在未经核对的情况下覆盖既有冻结版本。

PubChem 接口文档：<https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest>

PubChem 引用说明：<https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines>

## 2. PubChem 查询字段

PUG-REST 请求包含以下属性：

```text
Title
IUPACName
MolecularFormula
MolecularWeight
CanonicalSMILES
IsomericSMILES
InChI
InChIKey
ExactMass
MonoisotopicMass
TPSA
HBondDonorCount
HBondAcceptorCount
RotatableBondCount
XLogP
HeavyAtomCount
Charge
```

模型并不直接使用上面所有字段。保留完整响应是为了能够检查化合物身份、盐形式、立体化学和后续特征生成过程。实际进入模型的结构信息是由 Isomeric SMILES 计算得到的 Morgan 指纹和五项分子描述符。

## 3. 化合物名称与结构的对应

比赛 metadata 提供的是实验条件中的化合物名称。名称首先经过规范化，再映射到经过核对的 PubChem CID。质量控制标签 `Quality Control` 不视为化合物，也不参与结构特征构建。

冻结对照表覆盖 56 个化合物，其中：

- 39 个出现在训练划分；
- 45 个出现在验证划分；
- 56 个出现在测试 metadata；
- 6 个名称使用了显式 alias 规则。

这里的覆盖统计只依赖 metadata 中的化合物身份，没有读取蛋白质组真值，也没有计算来自验证集或测试集的目标统计量。

名称映射文件的 SHA-256 为：

```text
6def6b7f6624cdf5264dd7b1a6cf9a71e7aeafe88c9e2fe5209aa6bd451708a7
```

### 3.1 需要特别说明的代表结构

有三个名称不能简单理解为单一、完全确定的化学实体。项目使用了明确的代表结构，并保留这一选择，避免复现者在不知情的情况下得到不同特征。

| metadata 名称 | 本项目采用的表示 | 原因 |
|---|---|---|
| `1-10 Phenanthroline monohydrate` | 无水 1,10-phenanthroline 母体 | 结构特征使用母体分子，不把结晶水作为药效结构的一部分 |
| `Oligomycin` | Oligomycin A | metadata 没有指出具体同系物，采用可明确定位的代表成分 |
| `Tunicamycin` | Tunicamycin A | 名称可能指混合物或家族，metadata 不足以还原唯一组成 |

这些处理是可复现的建模约定，不代表不同同系物或混合物在化学和生物学上等价。如果后续取得主办方更精确的试剂信息，应保留现有版本，同时新增修订后的身份合同并重新生成特征。

### 3.2 盐、金属配合物和碎片处理

一般情况下，结构先经过 RDKit `FragmentParent` 处理，去除不作为主要分子骨架的小碎片或对离子。这样能够减少同一活性母体因盐形式不同而产生的不必要差异。

`Cisplatin` 和 `NaCl` 不适合按普通有机小分子的碎片规则处理，因此保留完整记录。该例外在特征合同中是显式规则，不依赖运行时猜测。

## 4. RDKit 结构标准化

特征由 RDKit `2026.3.5` 生成。处理顺序固定如下：

1. 从冻结的 PubChem 响应读取 Isomeric SMILES；
2. 使用 RDKit 解析分子；
3. 输出 RDKit canonical isomeric SMILES，作为规范化后的结构记录；
4. 默认使用 `rdMolStandardize.FragmentParent` 得到特征分子；
5. 对 `Cisplatin` 和 `NaCl` 应用完整记录例外；
6. 在同一特征分子上计算 Morgan 指纹和原始分子描述符；
7. 检查指纹维度、描述符有限性、CID 唯一性和实体顺序；
8. 写出冻结的特征数组、对照表和 manifest。

如果 RDKit 升级导致 canonical SMILES、FragmentParent 或描述符发生变化，应把它作为新的特征版本处理，不能静默替换当前快照。

## 5. Morgan 指纹

Morgan 指纹用于表达局部化学子结构。参数固定为：

| 参数 | 值 |
|---|---:|
| radius | 2 |
| diameter | 4 |
| fpSize | 2048 |
| includeChirality | `true` |
| useBondTypes | `true` |
| countSimulation | `false` |
| 输出类型 | 0/1 bit vector |
| 保存类型 | `uint8` |

每个化合物对应一个长度为 2,048 的二进制向量。实体顺序必须与化合物对照表一致，不能仅凭数组行号推断化合物身份。

## 6. 分子描述符

除 Morgan 指纹外，还计算五项连续描述符：

| 字段名 | RDKit 含义 |
|---|---|
| `mol_wt` | 分子量 |
| `mol_logp` | Crippen 方法估计的 LogP |
| `tpsa` | 拓扑极性表面积 |
| `hbd` | 氢键供体数量 |
| `hba` | 氢键受体数量 |

特征文件保存的是未经标准化的原始描述符。均值、标准差和缺失处理参数必须只用官方训练划分拟合，再应用到验证和测试 metadata。禁止用全部 56 个化合物共同估计缩放参数，因为这会让验证和测试实体参与训练统计量计算。

## 7. 模型如何使用结构特征

化合物输入由三部分组成：

- 训练集中出现过的化合物身份编码；
- 2,048 位 Morgan 指纹；
- 五项经过训练集参数标准化的分子描述符及其可用性标记。

对于训练中未出现、但能够从冻结合同取得结构的化合物，身份编码保持为空，结构指纹和描述符仍可使用。这样做的目的，是让模型在未知化合物场景下仍有可比较的化学信息，而不是把测试化合物当作训练类别加入词表。

结构特征不包含实验结果，也不携带蛋白质组标签。化合物是否出现在某个划分，仅用于检查输入覆盖情况，不参与任何目标值统计。

## 8. 冻结产物与校验值

| 产物 | SHA-256 |
|---|---|
| 化合物名称与 CID 对照表 | `6def6b7f6624cdf5264dd7b1a6cf9a71e7aeafe88c9e2fe5209aa6bd451708a7` |
| PubChem PUG-REST 原始响应 | `c1185c82564a1c699ca8941e5a47849834cfb6e4a351d8429cae02e1d5ccc24b` |
| 特征合同 | `cbf5f6a1b5feb0202d91b5e58c123bee89354b6e86b1e534b55b3a600fa8a327` |
| Morgan 指纹与描述符数组 | `d2cb4f5398182642209482624bec6c7f2da0fec9cd8425a26ac9638712f484df` |

生成或复制这些文件后，应运行：

```bash
sha256sum identifiers.csv pubchem-properties.json contract.csv fingerprints.npz
```

文件名可以根据仓库最终布局调整，但 manifest 中必须记录实际路径、实体数量、数组形状、RDKit 版本和对应校验值。

## 9. 与官方比赛数据的边界

主办方数据和 PubChem 数据需要严格分开管理：

- PubChem 结构数据是公开外部资源；
- 比赛 metadata 和蛋白质组矩阵由主办方提供，不因代码开源而自动获得再分发许可；
- 仓库不应包含测试集蛋白质组真值；
- 训练目标、蛋白筛选、类别词表、数值缩放和描述符缩放均只能使用官方训练划分拟合；
- 验证标签只用于允许范围内的早停和冻结后的评估；
- 测试推理只读取 metadata、模型参数和冻结的公开结构特征。

公开仓库可以提供数据目录结构、预期文件名、字段说明和校验脚本，但应要求使用者从主办方渠道自行取得受比赛规则约束的文件。

## 10. 未使用的外部信息

当前方案没有使用以下外部数据：

- 外部酵母蛋白质组或转录组标签；
- 验证集或测试集的蛋白质组真值特征；
- 预训练单细胞基础模型；
- 蛋白语言模型嵌入；
- Gene Ontology 注释；
- STRING 蛋白互作网络；
- 第三方药物响应标签。

如果未来引入新的外部资源，应单独记录数据库名称、版本或获取日期、许可证、实体映射方法、生成脚本、校验值，以及它是否会改变现有训练和评测边界。
