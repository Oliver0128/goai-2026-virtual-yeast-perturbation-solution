# 输入与输出合同

## 1. 推理输入

`code/submission/predict.py` 的必需参数为：

```text
--metadata             UTF-8 CSV，至少包含 sample_ID 和 checkpoint manifest 要求的字段
--structure-npz        冻结结构 NPZ
--structure-contract   与 NPZ 同轴的 contract.csv
--e7-model             E7 checkpoint
--cvae-model           CR-VAE checkpoint
--output               输出 CSV 路径
--audit                审计 JSON 路径
```

metadata 的实际必需字段是：

```text
sample_ID
Strains
perturbation_no_concentration
Medium
Temperature
pert_time
pert_time_unit
data_source
instrument
```

`sample_ID` 必须非空且唯一；`pert_time` 必须为有限正数。分类字段按 checkpoint manifest 的原始字符串匹配；未知分类会得到该 one-hot block 的全零编码。化合物名必须与 `contract.csv` 的 `competition_name` 或训练词表拼写一致；未知化合物会保留为零结构/身份特征。

完整官方 metadata 还可包含 `pert_id`、`Yeast_cell_plate`、`protein_well`、`split_final`、`strain_role`、`chemical_role`。这些额外列允许存在，普通推理不使用；只有显式传入 `--treatment-only` 时才会读取 `pert_id` 并执行官方提交合同要求的过滤。

## 2. 结构输入

`fingerprints.npz` 必须含有且只能含有以下五个数组：

```text
competition_names  (56,)         字符串
pubchem_cids       (56,)         整数
morgan_bits        (56, 2048)    uint8，只有 0/1
descriptor_names   (5,)          字符串
descriptors_raw    (56, 5)       有限浮点数
```

`contract.csv` 必须有 56 行，且 `competition_name`、`pubchem_cid` 的顺序与 NPZ 完全一致。五个 raw descriptor 的顺序由 `descriptor_names` 和 checkpoint contract 固定；运行时使用 checkpoint 中训练集拟合的均值/尺度，不重新拟合。

## 3. 中间表示和模型维度

当前正式 checkpoint 的接口为：

```text
E7 condition input     21 维
E7 drug input          2094 维
模型输出               4422 个保留蛋白
CR-VAE context         256 维
```

完整 feature names、类别列表、数值均值/尺度、保留蛋白顺序均嵌在 `models/e7/model.pt` 和 `models/crvae/model.pt` 的 manifest 中。不要从 CSV 列顺序猜测蛋白顺序或重新排序列。

## 4. 推理输出

输出 CSV 必须满足：

- UTF-8、逗号分隔、无额外索引列；
- 第一列为 `sample_ID`，顺序与输入 metadata 相同；
- 后续 4,422 列为 checkpoint `retained_proteins`，顺序固定；
- 所有预测值为有限浮点数。

审计 JSON 至少包含输入/权重/输出 SHA-256、`sample_count`、`protein_count`、`truth_loaded`、`posterior_called` 和 `inference_mode`。正式无真值推理必须满足 `truth_loaded=false`、`posterior_called=false`。

## 5. 训练输入与产出

训练 metadata 和 proteome 通过 `sample_ID` 一一对齐；proteome 第一列必须为 `sample_ID`，其余列为蛋白名称。训练脚本根据 `split_final` 建立 train/validation，不接受 test proteome。训练产出中的 `model.pt`、`manifest.json`、`config.snapshot.json`、`training-history.json`、验证预测和评分文件必须保存在同一个新输出目录中，便于 CR-VAE 绑定和审计。
