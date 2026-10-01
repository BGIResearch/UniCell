# UniCell

[English](README.md) | **简体中文**

UniCell 是一个面向单细胞转录组数据的本体（ontology）引导注释框架。当前实现以 H5AD 为输入，同时预测细胞类型、组织和物种，并可在推理阶段使用可选的分层 safe list，按照 `species -> tissue -> cell_type` 的顺序限制最终候选标签。

本仓库包含训练与推理代码、Python 包、Notebook、API/教程文档以及 wheel/离线发布辅助脚本。数据集、训练 checkpoint、推理结果和构建产物不直接提交到 Git。

> [!IMPORTANT]
> 当前 `pyproject.toml` 中的 Python **发行名**是 `unicell-fig5`，Python **导入名**是 `unicell`。项目正式占用并发布 `unicell` 发行名后，目标安装命令是 `pip install unicell`；在此之前，请使用本地 wheel、`pip install .`，或发布到包索引后的 `pip install unicell-fig5`。不要把导入名和发行名混为一谈。

## 目录

- [功能概览](#功能概览)
- [项目文件结构](#项目文件结构)
- [环境与安装](#环境与安装)
- [Checkpoint 文件](#checkpoint-文件)
- [H5AD 输入要求](#h5ad-输入要求)
- [Safe list：可选的候选标签约束](#safe-list可选的候选标签约束)
- [安装 Python 包后的使用方法](#安装-python-包后的使用方法)
- [直接运行仓库 Python 文件](#直接运行仓库-python-文件)
- [推理输出](#推理输出)
- [文档、Notebook 与发布说明](#文档notebook-与发布说明)

## 功能概览

- 同时预测 `cell_type`、`tissue` 和 `species`。
- 通过 Cell Ontology 图进行层级建模，并在结果中写出本体层级路径。
- 支持普通表达矩阵输入，并支持使用 GeneFormer 或 scGPT encoder 进行可配置的 UniCell 训练与推理。
- 支持单个 H5AD 推理、按样本分块的大文件推理、单卡训练和 `torchrun` DDP 训练。
- 支持只限制物种、只限制组织、只限制细胞类型，以及物种—组织—细胞类型的分层 safe list。
- 将预测标签、共享表征、三个分类头 logits 和类别顺序写回 AnnData。

## 项目文件结构

下列结构只展示 Git 仓库中与使用相关的主要文件；本地 `build/`、`dist/`、`*.egg-info/` 和 `__pycache__/` 等构建产物不属于源码结构。

```text
.
├── README.md / README_ZH.md
├── pyproject.toml / MANIFEST.in
├── train_Unicell.py
├── train_epoch_unicell.py
├── predict.py
├── predict_by_sample_chunks.py
├── run_umap.py
├── safe_list.json
├── predict.ipynb
├── predict_by_sample_chunks.ipynb
├── unicell-train-predicct.ipynb
├── unicell/
│   ├── anno_predict.py
│   ├── scDataset.py
│   ├── dataset.py
│   ├── hmcn.py
│   ├── loss.py
│   ├── trainer.py
│   ├── ontoGRAPH.py
│   ├── evaluate.py
│   ├── cli/train_backbone.py
│   ├── cl-basic.obo / graph.gml
│   ├── utils/
│   └── repo/
├── docs/
├── packaging/
├── demo/
└── THIRD_PARTY_NOTICES.md
```

### 根目录主要 Python 文件

| 文件 | 主要功能 | 安装 wheel 后的入口 |
|---|---|---|
| `train_Unicell.py` | 读取固定的训练集和验证集；建立训练基因顺序与本体；校验并对齐验证集；初始化单卡或 DDP 训练；保存最佳 checkpoint | `unicell-train` |
| `predict.py` | 对单个 H5AD 推理；写入三类预测、embedding、logits 和本体层级；如果存在真实标签则计算 Accuracy/Macro-F1 | `unicell-predict` |
| `predict_by_sample_chunks.py` | 以 backed 模式打开大 H5AD，按 `obs[sample_key]` 逐样本载入内存并输出 H5AD/CSV；支持断点跳过、压缩、逐块指标和汇总表 | `unicell-predict-chunks` |
| `train_epoch_unicell.py` | 每个 epoch 依次读取 `seed_i/train.h5ad`，但固定参考基因、本体、类别字典和验证集；用于特定的逐 epoch 换数据实验 | 仅源码运行 |
| `run_umap.py` | 先执行无 safe-list 的 UniCell 推理，再对共享表征和三个分类头输出计算 UMAP 并导出图片/H5AD | 仅源码运行 |

### `unicell/` 核心包

| 文件或目录 | 主要功能 |
|---|---|
| `anno_predict.py` | 加载 checkpoint、按训练基因顺序对齐输入、构建 HMCN、批量推理、执行可选 safe-list 解码，并把结果写回 AnnData |
| `scDataset.py` | 读取或封装 AnnData；构建训练所需的 Cell Ontology 子图；处理旧 ontology ID、细胞类型索引及 tissue/species 编码 |
| `dataset.py` | PyTorch Dataset 和数据转换层；支持表达矩阵、scFoundation、GeneFormer、scGPT 以及 LMDB 数据 |
| `hmcn.py` | 定义 Encoder、HMCN 全局/局部分支以及 cell type、tissue、species 三个分类头 |
| `loss.py` | 定义 HMCN 全局/局部层级损失、Focal Loss 和可选 MMD 批次损失 |
| `trainer.py` | 组装模型、DataLoader、损失与优化器；执行训练/验证；按最佳 F1 保存 checkpoint；记录训练耗时 |
| `ontoGRAPH.py` | 导入、裁剪和查询 Cell Ontology；生成层级映射与监督矩阵；序列化 checkpoint 的 ontology 制品 |
| `evaluate.py` | 基于 AnnData 标签列计算 Accuracy、Macro-F1 和 Micro-F1 |
| `utils/` | 指标、ontology、LMDB、日志、配置和兼容辅助函数 |
| `repo/` | GeneFormer、scGPT、scFoundation、scBERT 的适配或内嵌第三方代码；再分发前请阅读 `THIRD_PARTY_NOTICES.md` |
| `cli/train_backbone.py` | wheel 内的可配置训练入口，负责 GeneFormer/scGPT/expr 校验和可迁移 checkpoint 生成 |

其他目录：

- `docs/`：API、模型结构与教程文档。
- `packaging/`：wheel 构建、多架构 release、离线 wheelhouse、安装和 smoke test 脚本。
- `demo/`：人和小鼠 atlas 的 Cell Ontology ID 参考清单；不包含可直接运行的 H5AD 或 checkpoint。
- 根目录三个 Notebook：标准推理、分块推理和训练/推理交互式示例。

## 环境与安装

### 基本要求

- Linux
- Python `>=3.9,<3.10`，即当前版本要求 Python 3.9
- PyTorch 2.0.x
- GPU 推理/训练需要与 PyTorch 匹配的 CUDA 环境；也可使用 CPU，但速度通常明显更慢

依赖和 x86_64/aarch64 离线发布方法见 [`packaging/README.md`](packaging/README.md) 和 [`packaging/USAGE_ZH.md`](packaging/USAGE_ZH.md)。

### 从包索引安装

项目正式以 `unicell` 发行名发布后，期望用户通过下面的命令下载并安装官方 wheel：

```bash
python -m pip install unicell
```

但**当前仓库元数据尚未使用该发行名**。目前 `pyproject.toml` 的发行名为 `unicell-fig5`，因此当前准确的索引安装命令应为：

```bash
# 仅在 unicell-fig5 已发布到所使用的 PyPI/私有索引后有效
python -m pip install unicell-fig5
```

若要正式启用 `pip install unicell`，发布者需要先确认该包索引名称归本项目所有，将 `[project].name` 改为 `unicell`，重新构建 wheel，并将新 wheel 发布到对应索引。仅修改 README 不能改变 wheel 的安装名称。

### 安装本地 wheel

```bash
python -m pip install /path/to/unicell_fig5-0.2.1-py3-none-any.whl
```

wheel 包含 Python 包、四个 console command、本体资源和轻量词表，但**不包含** UniCell checkpoint 或 GeneFormer/scGPT 预训练权重，也不包含 `safe_list.json`、`run_umap.py` 或 `train_epoch_unicell.py`。训练时需要提供本地基础模型文件；由 `unicell-train-backbone` 生成的完整 portable checkpoint 在推理时不再依赖这些原始路径。

### 从源码安装

```bash
git clone https://github.com/luyi98/unicell.git
cd unicell

python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

开发模式安装：

```bash
python -m pip install -e .
```

验证安装：

```bash
python -c "import unicell; print(unicell.__version__)"
unicell-predict --help
unicell-predict-chunks --help
unicell-train --help
unicell-train-backbone --help
```

## Checkpoint 文件

模型文件不提交到 Git。v0.2.1 Release 提供带版本号的 expression checkpoint
及其完整性元数据：

- [下载 UniCell expression checkpoint](https://github.com/luyi98/unicell/releases/download/v0.2.0/unicell-last_version_gml-v1.tar.gz)
- [下载 checkpoint manifest](https://github.com/luyi98/unicell/releases/download/v0.2.1/checkpoint-manifest.json)
- [下载 SHA256 校验文件](https://github.com/luyi98/unicell/releases/download/v0.2.1/SHA256SUMS)

checkpoint 压缩包直接复用 v0.2.0 的下载链接，不重复上传；manifest 保持不变，其中保留原始训练和源码来源信息。代码改动与验证范围见 [v0.2.1 发布说明](docs/RELEASE_NOTES_0.2.1.md)。

### 基础 checkpoint 文件

```text
checkpoint_directory/
├── unicell_v1.best.pth
├── gene_names.pk
├── ontoGraph.pk
├── ontoGraph.graph.gml
├── celltype_dict.pk
├── tissue_dict.pk
└── species_dict.pk
```

| 文件 | 作用 |
|---|---|
| `unicell_v1.best.pth` | PyTorch checkpoint，包含 `model_state_dict` 和模型结构 metadata，例如输入类型、输入/输出维度、global/local layer、dropout 以及外部基础模型路径 |
| `gene_names.pk` | 训练基因的有序列表；推理按该顺序重排输入，忽略额外基因，并为 checkpoint 中存在但输入中缺失的基因补 0 |
| `ontoGraph.pk` | 序列化的 `OntoGRAPH` 对象，保存 vocabulary、层级、hierarchical arrays 和类别索引；保存时对象中的实际 graph 被置空 |
| `ontoGraph.graph.gml` | 与 `ontoGraph.pk` 配套的实际 ontology 子图；加载时重新挂回对象 |
| `celltype_dict.pk` | 细胞类型标签到 flat cell-type 分类头索引的映射；当前标签通常是 Cell Ontology ID |
| `tissue_dict.pk` | 组织名称到 tissue 分类头索引的映射 |
| `species_dict.pk` | 物种名称到 species 分类头索引的映射 |

注意事项：

- 七个文件必须来自同一次兼容的训练发布，不能混用不同版本。分类头维度、字典、基因顺序和 ontology 必须一致。
- 包内的 `unicell/graph.gml` 用于训练时构造 ontology，不等同于 checkpoint 中的 `ontoGraph.graph.gml`。
- `train_time_summary.json`、`timing_logs/` 是训练日志，不是推理必需文件。
- `unicell_v1.ep*.pth` 只保存 raw state dict，不含完整 metadata，不能直接替代 `unicell_v1.best.pth`。
- 对 `input_type="expr"` 模型，上述七件套是完整推理制品。
- `.pth` 和 `.pk` 使用 Python/PyTorch 反序列化机制，可能执行恶意代码。只加载可信来源的 checkpoint，并在加载前校验 SHA256。

### Portable GeneFormer/scGPT checkpoint

由 `unicell-train-backbone` 生成的 foundation-backed checkpoint 还会包含
`run_config.json` 和轻量重建 bundle：

```text
checkpoint_directory/
├── 上述七个基础文件
├── run_config.json
└── backbone/
    ├── manifest.json
    ├── config.json + gene_vocab.json    # GeneFormer
    └── args.json + vocab.json           # scGPT
```

目录中只会出现所选 backbone 对应的一组文件。训练后的 backbone 参数位于
`unicell_v1.best.pth`；`backbone/` 只保存重建网络结构所需的小型配置和词表，
不复制原始 GeneFormer/scGPT 预训练权重。请整体移动 checkpoint 目录，并将
新位置传给 `--ckpt_dir`。旧 checkpoint 可能仍保存绝对的
`llm_model_file`、`llm_vocab_file` 和 `llm_args_file` 路径，可用下文的推理参数覆盖。

## H5AD 输入要求

### 推理数据

- 输入必须是可由 `anndata.read_h5ad()` 读取的 `.h5ad` 文件。
- `adata.var_names` 应使用与 `gene_names.pk` 一致的基因标识方式，通常是基因符号；至少一个基因必须与 checkpoint 匹配，否则推理会报错。
- 推理会按 checkpoint 的基因顺序重建表达矩阵：额外输入基因被忽略，缺失基因补 0。
- `unicell_predict()` 会复制输入，并执行与训练/验证读取器一致的启发式预处理。对于基因数大于 1,000 的非负矩阵，会过滤检测基因少于 200 的细胞；若最大值大于 25，则执行 `log1p`，其中看起来是整数 counts 的输入会先执行 `normalize_total(1e4)`。已经归一化并 log1p、且最大值不超过 25 的数据不会再次归一化或 log1p，但仍会执行上述细胞过滤。
- 如果只做预测，`obs` 中不要求存在真实 cell type/tissue/species 列；缺少这些列时只会跳过相应指标。
- 如果希望计算指标，默认真实标签列为：
  - `cell_type_ontology_term_id`
  - `general_tissue`
  - `organism`
- 分块推理还要求 `obs` 中存在由 `--sample_key` 指定的列，默认是 `sample`。

> [!CAUTION]
> 推理会创建一个按 checkpoint 基因重排的新 AnnData。原始 `var`、`layers`、`raw` 等内容不会被完整保留。若这些内容后续仍需使用，请先保留原始 H5AD。

### 训练与验证数据

默认训练入口要求训练集和验证集的 `obs` 都包含：

```text
cell_type_ontology_term_id
general_tissue
organism
```

此外：

- 细胞类型应使用模型 ontology 能识别的 Cell Ontology ID。
- 验证集必须包含训练集所需的全部基因；代码会将验证集重排为训练基因顺序。
- 验证集不能含训练集中从未出现的 tissue 或 species 标签。
- 当前训练读取流程中的 HVG 选择代码被关闭，不应假定程序会自动完成 HVG 筛选。
- 训练入口的 `scDataset.read_data()` 仍带有基于表达值范围的启发式处理：对非负表达矩阵，在基因数大于 1000 时过滤检测基因数少于 200 的细胞，并可能对看起来像原始计数的数据执行 `normalize_total(1e4)` 和 `log1p`。发布 checkpoint 时应记录实际触发的预处理。
- 建议在训练前保证 `obs_names` 和 `var_names` 唯一，并显式记录归一化、log1p 和基因筛选流程。

## Safe list：可选的候选标签约束

Safe list 只作用于**推理阶段的最终标签解码**，不参与训练，也不会修改保存到 `obsm` 中的原始 logits。未使用 safe list 时，物种、组织和细胞类型三个分类头各自执行全类别 `argmax`；使用后逐细胞按照下面的顺序解码：

```text
species -> tissue -> cell_type
```

`predict.py` 和 `predict_by_sample_chunks.py` 都调用同一个 `unicell_predict()`，所以 safe-list 规则完全相同。

### 是否可以不传或留空

Safe list 不是必需文件。推荐的无约束方式是：

- CLI 中完全省略 `--safe_list`。
- Python API 中设置 `safe_list_path=None`。

合法 JSON `{}` 或下面的内容当前也等价于无约束：

```json
{
  "global": {},
  "hierarchy": {}
}
```

不要传零字节空文件，因为空文件不是合法 JSON，会触发 `JSONDecodeError`。也不要用空数组 `[]` 表示“禁止全部”；当前实现不支持 deny-all，空候选集会回退到无约束 `argmax`。

### 完整 JSON 格式

仓库中的 [`safe_list.json`](safe_list.json) 是一个格式示例：

```json
{
  "global": {
    "species": ["Homo sapiens", "Mus musculus"],
    "tissue": ["brain", "lung", "heart"],
    "cell_type": ["CL:0000540", "CL:0000127", "CL:0000236"]
  },
  "hierarchy": {
    "Homo sapiens": {
      "tissue": {
        "brain": {
          "cell_type": ["CL:0000540", "CL:0000127"]
        },
        "lung": {
          "cell_type": ["CL:0000236"]
        }
      }
    },
    "Mus musculus": {
      "tissue": {
        "brain": {
          "cell_type": ["CL:0000540"]
        }
      }
    }
  }
}
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `global.species` | 字符串数组 | 全局允许的物种候选 |
| `global.tissue` | 字符串数组 | 全局允许的组织候选 |
| `global.cell_type` | 字符串数组 | 全局允许的细胞类型候选 |
| `hierarchy.<species>.tissue` | JSON object | 已预测为该物种时允许的组织；object 的键就是组织名 |
| `hierarchy.<species>.tissue.<tissue>.cell_type` | 字符串数组 | 已预测为该“物种 + 组织”组合时允许的细胞类型 |

顶层必须是 JSON object；`global` 和 `hierarchy` 都可省略。所有名称必须与当前 checkpoint 字典的键**完全一致**，包括大小写和空格：

- 物种查 `species_dict.pk`。
- 组织查 `tissue_dict.pk`。
- 细胞类型查 `celltype_dict.pk`，通常应填写 `CL:...` ID，而不是自然语言展示名称。
- safe-list 字段名固定为 `species`；这与 H5AD 默认物种列名 `organism` 是两个不同概念。

### 只限制一个维度

只限制物种：

```json
{
  "global": {
    "species": ["Homo sapiens"]
  }
}
```

只限制组织：

```json
{
  "global": {
    "tissue": ["brain", "lung"]
  }
}
```

只限制细胞类型：

```json
{
  "global": {
    "cell_type": ["CL:0000540", "CL:0000127"]
  }
}
```

没有出现的维度继续在 checkpoint 的全部类别中选择。例如，只写 `global.tissue` 不会限制物种或细胞类型，也不会自动推断它们与组织的生物学关系。

### 分层限制

上面的完整示例表示：

1. 先在 `global.species` 中选择物种。
2. 如果预测为 `Homo sapiens`，组织只能在 `brain`、`lung` 中选择。
3. 如果预测为 `Mus musculus`，组织只能选择 `brain`。
4. 最后从相应“物种 + 组织”叶节点的 `cell_type` 中选择细胞类型。

`hierarchy` 中出现某个物种**不会限制物种分类头**；物种层只读取 `global.species`。因此，若要进行严格的分层限制，应同时：

- 在非空的 `global.species` 中写出全部允许物种。
- 为每个允许物种配置至少一个 checkpoint 可识别的组织。
- 为每个需要限制的物种—组织组合配置至少一个 checkpoint 可识别的细胞类型。

### `global` 与 `hierarchy` 如何组合

物种层只使用：

```text
global.species
```

组织层在两侧约束都存在时使用：

```text
global.tissue ∩ hierarchy[已预测物种].tissue.keys()
```

细胞类型层在两侧约束都存在时使用：

```text
global.cell_type
∩ hierarchy[已预测物种].tissue[已预测组织].cell_type
```

组合规则：

- 两侧都存在时取交集。
- 只有一侧存在时使用该侧。
- 两侧都不存在时该层不限制。
- 如果已预测物种没有 hierarchy 分支，组织层只应用仍存在的 global 约束。
- 如果某个“物种 + 组织”没有 `cell_type` 配置，细胞类型层只应用仍存在的 global 约束。

例如，仓库示例的 `global.tissue` 中虽然包含 `heart`，但两个 hierarchy 物种分支中都没有 `heart`，因此两者取交集后不会允许 `heart`。

### 空值、未知标签和冲突的当前行为

| 配置情况 | 当前实现行为 |
|---|---|
| 字段省略 | 该来源不施加限制 |
| `{}`、空 `global` 或空 `hierarchy` | 相应部分不限制 |
| 数组中同时有已知和未知标签 | 打印 `[SafeList][WARN]`；已知标签继续生效 |
| 数组为 `[]` | 空候选集最终回退到该分类头的全类别 `argmax` |
| 数组中的标签全部未知 | 警告后得到空候选集，再回退到全类别 `argmax` |
| global 与 hierarchy 的交集为空 | 回退到全类别 `argmax`，结果甚至可能同时位于两份列表之外 |
| hierarchy 某物种没有任何有效组织 | 不建立该物种的组织限制 |
| 某组织省略 `cell_type` 或将其设为 `null` | 该物种—组织组合不施加分层细胞类型限制 |
| JSON 类型错误 | 可能直接触发 `TypeError` 或 `AttributeError` |

因此，发布 safe list 前应检查：

- 不使用空数组表达拒绝全部。
- 每个数组至少包含一个对应 checkpoint 已知标签。
- 所有需要同时生效的 global/hierarchy 交集都非空。
- 数组必须写成 JSON array，不能把单个值写成普通字符串。

可以用下面的脚本查看 checkpoint 支持的精确标签。pickle 文件同样只能从可信来源加载：

```python
from pathlib import Path
import pickle

ckpt_dir = Path("/path/to/checkpoint")

for filename in (
    "species_dict.pk",
    "tissue_dict.pk",
    "celltype_dict.pk",
):
    with (ckpt_dir / filename).open("rb") as handle:
        label_to_index = pickle.load(handle)
    print(filename, list(label_to_index.keys()))
```

## 安装 Python 包后的使用方法

安装 wheel 或源码包后可获得四个命令：

| 命令 | 用途 |
|---|---|
| `unicell-predict` | 单个 H5AD 推理 |
| `unicell-predict-chunks` | 按样本分块推理大型 H5AD |
| `unicell-train` | 原有普通表达矩阵训练流程的兼容入口 |
| `unicell-train-backbone` | 使用 expr、GeneFormer 或 scGPT encoder 的可校验、可配置训练入口 |

### 单文件推理 CLI

```bash
unicell-predict \
  --input ./data/input.h5ad \
  --ckpt_dir ./models/unicell_expression_v1 \
  --output ./results/input_annotated.h5ad \
  --device cuda \
  --batch_size 2048 \
  --safe_list ./safe_list.json \
  --save_metrics_json
```

如果不需要约束，请直接删除 `--safe_list`。参数如下：

| 参数 | 必填/默认值 | 说明 |
|---|---|---|
| `--input` | 必填 | 输入 `.h5ad` 文件 |
| `--ckpt_dir` | 必填 | 包含七个必需文件的 checkpoint 目录 |
| `--output` | `<input>_annotated.h5ad` | 输出 H5AD 路径 |
| `--device` | 自动选择 | 可选 `cuda` 或 `cpu`；不传时有 CUDA 就用 CUDA，否则用 CPU |
| `--batch_size` | `2048` | 推理 batch size；显存不足时减小 |
| `--safe_list` | 不使用 | 可选 safe-list JSON 路径 |
| `--backbone_model` | checkpoint metadata | 覆盖旧 GeneFormer 模型目录或 scGPT `.pt` 路径 |
| `--backbone_vocab` | checkpoint metadata | 覆盖旧 GeneFormer/scGPT 词表路径 |
| `--backbone_args` | checkpoint metadata | 覆盖旧 scGPT `args.json` 路径 |
| `--init_backbone_from_config` | checkpoint metadata | 先按配置重建 backbone，再加载完整 UniCell state dict；主要用于迁移 |
| `--save_metrics_json` | 关闭 | 将指标保存到 `<output_stem>_metrics.json` |
| `--cell_type_key` | `cell_type_ontology_term_id` | `obs` 中真实细胞类型列，用于指标计算 |
| `--tissue_key` | `general_tissue` | `obs` 中真实组织列，用于指标计算 |
| `--species_key` | `organism` | `obs` 中真实物种列，用于指标计算 |

只要真实标签列存在，命令就会在终端打印相应指标；`--save_metrics_json` 只控制是否额外保存 JSON。

### 大文件按样本分块推理 CLI

```bash
unicell-predict-chunks \
  --input ./data/atlas.h5ad \
  --ckpt_dir ./models/unicell_expression_v1 \
  --out_dir ./results/atlas_chunks \
  --sample_key sample \
  --device cuda \
  --batch_size 2048 \
  --compression gzip \
  --compression_opts 4 \
  --skip_done \
  --save_metrics_json
```

| 参数 | 必填/默认值 | 说明 |
|---|---|---|
| `--input` | 必填 | 输入大型 `.h5ad` 文件 |
| `--ckpt_dir` | 必填 | checkpoint 目录 |
| `--out_dir` | 必填 | 每个样本的 H5AD/CSV 输出目录 |
| `--sample_key` | `sample` | 用于拆分的 `obs` 列 |
| `--device` | `cuda` | `cuda` 或 `cpu`；请求 CUDA 但不可用时自动回退 CPU |
| `--batch_size` | `2048` | 每个分块内部的推理 batch size |
| `--safe_list` | 不使用 | 可选 safe-list JSON 路径 |
| `--backbone_model` | checkpoint metadata | 覆盖旧 GeneFormer 模型目录或 scGPT `.pt` 路径 |
| `--backbone_vocab` | checkpoint metadata | 覆盖旧 GeneFormer/scGPT 词表路径 |
| `--backbone_args` | checkpoint metadata | 覆盖旧 scGPT `args.json` 路径 |
| `--init_backbone_from_config` | checkpoint metadata | 先按配置重建 backbone，再加载完整 UniCell state dict；主要用于迁移 |
| `--compression` | `gzip` | H5AD 压缩方式：`gzip` 或 `none` |
| `--compression_opts` | `4` | gzip 压缩等级 |
| `--skip_done` | 关闭 | 同一块的 H5AD 和 CSV 都已存在时跳过 |
| `--save_metrics_json` | 关闭 | 为每个分块保存指标 JSON |
| `--summary_csv` | `<out_dir>/chunk_summary.csv` | 自定义汇总 CSV 路径 |
| `--cell_type_key` | `cell_type_ontology_term_id` | `obs` 中真实细胞类型列 |
| `--tissue_key` | `general_tissue` | `obs` 中真实组织列 |
| `--species_key` | `organism` | `obs` 中真实物种列 |

每个样本会输出一个带注释的 H5AD 和一个预测 CSV，任务结束后另写汇总 CSV。

### 原有普通表达矩阵训练 CLI

`unicell-train` 保留固定 `input_type="expr"` 的原有流程。可配置的普通表达矩阵、
GeneFormer 或 scGPT 训练请使用下文的 `unicell-train-backbone`；新命令不会改变旧入口。

单卡或自动选择 CPU/GPU：

```bash
unicell-train \
  --train_h5ad ./data/train.h5ad \
  --eval_h5ad ./data/valid.h5ad
```

多 GPU DDP：

```bash
torchrun --standalone --nproc_per_node=2 --module train_Unicell \
  --train_h5ad ./data/train.h5ad \
  --eval_h5ad ./data/valid.h5ad
```

当前 `unicell-train` 只有两个 CLI 参数：

| 参数 | 必填 | 说明 |
|---|---|---|
| `--train_h5ad` | 是 | 训练集 H5AD |
| `--eval_h5ad` | 是 | 验证集 H5AD |

其余设置不是 CLI 参数。在 `train_Unicell.py` 中，下表的大写项目是模块级常量，而 `input_type` 和 `output_dim` 是构造 Trainer 时写死的参数：

| 配置 | 当前值 | 作用 |
|---|---|---|
| `PREFIX` | `checkpoints` | 实验/输出目录后缀 |
| `CKPT_DIR` | `models/checkpoints` | checkpoint 输出目录，相对于当前工作目录 |
| `BATCH_SIZE` | `128` | 训练 batch size |
| `LEARNING_RATE` | `1e-3` | Adam 学习率 |
| `NUM_EPOCHS` | `50` | epoch 数 |
| `BETA` | `0.1` | Trainer 中组合损失的权重参数 |
| `GLOBAL_LAYER` | `128` | HMCN global hidden size |
| `LOCAL_LAYER` | `64` | HMCN local hidden size |
| `HIDDEN_LAYER_DROPOUT` | `0.1` | 隐藏层 dropout |
| `CELL_TYPE_KEY` | `cell_type_ontology_term_id` | 细胞类型标签列 |
| `TISSUE_KEY` | `general_tissue` | 组织标签列 |
| `SPECIES_KEY` | `organism` | 物种标签列 |
| `input_type` | `expr` | 当前入口使用普通表达矩阵 |
| `output_dim` | `512` | Encoder 输出维度 |

若要自定义大写项目，可以修改源码常量后重新安装，也可以按下面的 Python 示例在调用前覆盖模块配置。修改 `input_type` 或 `output_dim` 则需要编辑 Trainer 的构造调用，或直接使用底层 `UnicellTrainer` API。

### Foundation encoder 训练 CLI（GeneFormer/scGPT）

0.2.0 wheel 新增的 <code>unicell-train-backbone</code> 是可配置训练入口。训练集和验证集都必须包含 <code>cell_type_ontology_term_id</code>、<code>general_tissue</code> 和 <code>organism</code>。命令启动时总会执行检查；添加 <code>--validate_only</code> 可以只检查文件路径、H5AD schema、基因重叠、词表、模型权重结构、设备和每个 DDP rank 的细胞数，而不开始训练。

GeneFormer 默认使用 wheel 内置词表。<code>--llm_model_file</code> 应指向包含 <code>config.json</code> 和本地权重的 Hugging Face 模型目录：

```bash
unicell-train-backbone \
  --input_type GeneFormer \
  --train_h5ad ./data/train.h5ad \
  --eval_h5ad ./data/valid.h5ad \
  --llm_model_file /models/Geneformer-V1-10M \
  --ckpt_dir ./models/unicell_geneformer \
  --validate_only
```

scGPT 的模型、词表和 <code>args.json</code> 必须来自同一个预训练发布：

```bash
unicell-train-backbone \
  --input_type scGPT \
  --train_h5ad ./data/train.h5ad \
  --eval_h5ad ./data/valid.h5ad \
  --llm_model_file /models/scgpt/best_model.pt \
  --llm_vocab_file /models/scgpt/vocab.json \
  --llm_args_file /models/scgpt/args.json \
  --ckpt_dir ./models/unicell_scgpt \
  --validate_only
```

检查通过后删除 <code>--validate_only</code> 即可训练。多 GPU 使用 <code>torchrun --standalone --nproc_per_node=N -m unicell.cli.train_backbone</code> 并追加相同参数。现有 Trainer 使用 <code>drop_last=True</code>，所以每个进程至少要分到一个完整 batch。

wheel 不分发 foundation 模型权重；训练时从上述外部路径载入。成功训练后会写出
七个基础文件、`run_config.json`，以及 [checkpoint 文件](#checkpoint-文件)中说明的
portable `backbone/` bundle。整体移动 checkpoint 目录后，不需要原始预训练模型路径即可推理：

```bash
unicell-predict \
  --input ./data/test.h5ad \
  --ckpt_dir ./models/moved_unicell_scgpt \
  --output ./results/test_annotated.h5ad \
  --device cuda
```

对于 metadata 仍指向另一台机器绝对路径的旧 GeneFormer/scGPT checkpoint，可按下面方式覆盖：

```bash
# 旧 GeneFormer checkpoint
unicell-predict --input ./data/test.h5ad \
  --ckpt_dir ./models/legacy_geneformer \
  --backbone_model /models/Geneformer-V1-10M \
  --output ./results/geneformer_annotated.h5ad

# 旧 scGPT checkpoint
unicell-predict --input ./data/test.h5ad \
  --ckpt_dir ./models/legacy_scgpt \
  --backbone_model /models/scgpt/best_model.pt \
  --backbone_vocab /models/scgpt/vocab.json \
  --backbone_args /models/scgpt/args.json \
  --output ./results/scgpt_annotated.h5ad
```

`--init_backbone_from_config` 是迁移用高级选项：先按配置重建网络，再加载完整且兼容的
UniCell state dict。`unicell-predict-chunks` 支持相同的四个参数。标准
`input_type="expr"` checkpoint 不会进入 foundation encoder 分支。

### Python API 推理

```python
from pathlib import Path

import anndata as ad
import torch

from unicell.anno_predict import unicell_predict

input_path = Path("./data/input.h5ad")
output_path = Path("./results/input_annotated.h5ad")
adata = ad.read_h5ad(input_path)

result = unicell_predict(
    adata=adata,
    filepath=str(input_path),
    ckpt_dir="./models/unicell_expression_v1",
    batch_size=512,
    device="cuda" if torch.cuda.is_available() else "cpu",
    cell_type_key="cell_type_ontology_term_id",
    tissue_key="general_tissue",
    species_key="organism",
    compute_metrics=False,
    safe_list_path=None,
)

output_path.parent.mkdir(parents=True, exist_ok=True)
result.adata.write_h5ad(output_path)
```

返回值是 `scDataset`，不是直接返回 `AnnData`；带注释的对象位于 `result.adata`。

`unicell_predict()` 参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `adata` | 无 | 必填的 AnnData 对象 |
| `filepath` | 无 | 必填的输入文件标识/路径；通常传入读取该 AnnData 的 H5AD 路径 |
| `ckpt_dir` | 语法默认 `None`，实际必填 | checkpoint 目录；不传会在拼接模型路径时报错 |
| `batch_size` | `512` | Python API 推理 batch size |
| `device` | `None` | `None` 时自动选择 CUDA/CPU，也可显式传 `"cuda"` 或 `"cpu"` |
| `cell_type_key` | `cell_type_ontology_term_id` | 指标使用的真实细胞类型列 |
| `tissue_key` | `general_tissue` | 指标使用的真实组织列 |
| `species_key` | `organism` | 指标使用的真实物种列 |
| `compute_metrics` | `True` | 对存在的真实标签列计算指标；缺失的任务会跳过 |
| `safe_list_path` | `None` | safe-list JSON 路径；`None` 表示无约束 |
| `backbone_model_file` | `None` | 覆盖旧 checkpoint 中记录的 foundation 模型路径 |
| `backbone_vocab_file` | `None` | 覆盖旧 checkpoint 中记录的 foundation 词表路径 |
| `backbone_args_file` | `None` | 覆盖旧 scGPT `args.json` 路径 |
| `initialize_backbone_from_config` | checkpoint metadata | 先按配置重建 foundation 结构，再加载完整 UniCell state dict |

### Python API 训练

wheel 会安装顶层模块 `train_Unicell`，可直接调用已封装的数据校验和训练流程：

```python
import torch
import train_Unicell as train_entry

# 可选：在调用前覆盖模块级配置。
train_entry.CKPT_DIR = "./models/experiment_a"
train_entry.BATCH_SIZE = 128
train_entry.LEARNING_RATE = 1e-3
train_entry.NUM_EPOCHS = 50

train_entry.train_unicell(
    ddp_train=False,
    local_rank=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    train_h5ad="./data/train.h5ad",
    eval_h5ad="./data/valid.h5ad",
)
```

`train_unicell()` 的直接参数只有：

| 参数 | 说明 |
|---|---|
| `ddp_train` | 是否按 DDP 模式训练；普通 Python 调用使用 `False` |
| `local_rank` | 本地进程/GPU rank；普通 Python 调用使用 `0` |
| `device` | 例如 `cuda`、`cuda:0` 或 `cpu` |
| `train_h5ad` | 训练集路径 |
| `eval_h5ad` | 验证集路径 |

模块常量是当前入口的配置方式，并非稳定的高层超参数 API。需要完全控制模型时，可直接使用 `unicell.trainer.UnicellTrainer`。其构造参数包括：

```text
scDataset, input_type, input_dim, output_dim,
batch_size, learning_rate, num_epochs, beta, device,
global_layer, local_layer, hidden_layer_dropout, ckpt_dir,
ddp_train=False, save_epoch=False, local_rank=0,
llm_model_file=None, llm_vocab_file=None, llm_args_file=None
```

低层 API 要求调用方自行正确构建/对齐 `scDataset`，并保存 `gene_names.pk`、`ontoGraph.pk` 和 `ontoGraph.graph.gml`。`UnicellTrainer` 只会补充模型权重和三个标签字典；如果遗漏 ontology 或基因文件，生成目录不能直接用于推理。优先使用 `train_Unicell.train_unicell()`，除非确实需要自定义底层训练流程。

## 直接运行仓库 Python 文件

在仓库根目录安装依赖后，核心脚本与安装后的命令对应：

```bash
# 参数与 unicell-predict 完全相同
python predict.py \
  --input ./data/input.h5ad \
  --ckpt_dir ./models/unicell_expression_v1 \
  --output ./results/input_annotated.h5ad \
  --device cuda \
  --batch_size 2048 \
  --safe_list ./safe_list.json \
  --save_metrics_json

# 参数与 unicell-predict-chunks 完全相同
python predict_by_sample_chunks.py \
  --input ./data/atlas.h5ad \
  --ckpt_dir ./models/unicell_expression_v1 \
  --out_dir ./results/atlas_chunks \
  --sample_key sample \
  --skip_done

# 参数与 unicell-train 完全相同
python train_Unicell.py \
  --train_h5ad ./data/train.h5ad \
  --eval_h5ad ./data/valid.h5ad

# 从源码运行可配置的 expr/GeneFormer/scGPT 训练
python -m unicell.cli.train_backbone --help
```

源码多 GPU：

```bash
torchrun --standalone --nproc_per_node=2 train_Unicell.py \
  --train_h5ad ./data/train.h5ad \
  --eval_h5ad ./data/valid.h5ad
```

完整参数与上一节对应的表格一致，也可以直接查看：

```bash
python predict.py --help
python predict_by_sample_chunks.py --help
python train_Unicell.py --help
python -m unicell.cli.train_backbone --help
```

### 每个 epoch 更换训练集

目录需要组织为：

```text
seed_root/
├── seed_0/train.h5ad
├── seed_1/train.h5ad
├── ...
└── seed_49/train.h5ad
```

运行：

```bash
python train_epoch_unicell.py \
  --eval_h5ad ./data/valid.h5ad \
  --seed_root ./data/sampled_h5ads \
  --seed_start 0
```

| 参数 | 必填/默认值 | 说明 |
|---|---|---|
| `--eval_h5ad` | 必填 | 所有 epoch 共用的验证集 |
| `--seed_root` | 必填 | 包含 `seed_i/train.h5ad` 的根目录 |
| `--seed_start` | `0` | 第一个 epoch 的 seed 编号；第 `e` 个 epoch 使用 `seed_start + e` |

该脚本固定运行 50 个 epoch，依次读取 `<seed_root>/seed_<seed_start>/train.h5ad` 到 `<seed_root>/seed_<seed_start+49>/train.h5ad`，输出目录固定为 `models/last_version_gml`。它不是 wheel console command；修改 epoch 数或输出目录需要编辑脚本顶部常量。

### 推理并绘制 UMAP

```bash
python run_umap.py \
  --test_h5ad ./data/test.h5ad \
  --ckpt_dir ./models/unicell_expression_v1 \
  --out_dir ./results/unicell_umap \
  --device cuda \
  --batch_size 2048 \
  --n-neighbors 30 \
  --min-dist 0.5 \
  --legend
```

| 参数 | 必填/默认值 | 说明 |
|---|---|---|
| `--test_h5ad` | 必填 | 输入 H5AD |
| `--ckpt_dir` | 必填 | checkpoint 目录 |
| `--out_dir` | `./unicell_umap` | 图片、元数据和可选 H5AD 输出目录 |
| `--species_col` | `organism` | 物种着色/指标列 |
| `--tissue_col` | `general_tissue` | 组织着色/指标列 |
| `--celltype_col` | `cell_type_ontology_term_id` | 细胞类型着色/指标列 |
| `--shared_emb_key` | `unicell_emb` | 共享 embedding 的 `obsm` key |
| `--species_head_key` | `species_cls_emb` | species logits 的 `obsm` key |
| `--tissue_head_key` | `tissue_cls_emb` | tissue logits 的 `obsm` key |
| `--celltype_head_key` | `cls_emb` | cell-type logits 的 `obsm` key |
| `--device` | 自动选择 | `cuda` 或 `cpu` |
| `--batch_size` | `2048` | 推理 batch size |
| `--no_metrics` | 关闭 | 传入后关闭 `unicell_predict()` 内部指标 |
| `--n-neighbors` | `30` | UMAP 邻居数 |
| `--min-dist` | `0.5` | UMAP minimum distance |
| `--random-state` | `0` | 随机种子 |
| `--legend` | 关闭 | 在图中显示图例 |
| `--pt-size` | `4.0` | 散点大小 |
| `--alpha` | `0.8` | 散点透明度 |
| `--no_write_h5ad` | 关闭 | 传入后不保存带 embedding/UMAP 的 H5AD |

`run_umap.py` 当前没有 `--safe_list` 参数，因此脚本内部执行的是无约束推理。

## 推理输出

标准推理会在结果 AnnData 中写入：

| 位置 | Key | 内容 |
|---|---|---|
| `obs` | `predicted_cell_type_ontology_id` | 预测的 Cell Ontology ID |
| `obs` | `predicted_cell_type` | 当前与预测 ontology ID 相同的字符串表示 |
| `obs` | `predicted_tissue` | 预测组织 |
| `obs` | `predicted_species` | 预测物种 |
| `obs` | `level_0`, `level_1`, ... | 预测细胞类型从共同祖先到叶节点的 ontology 路径 |
| `obsm` | `unicell_emb` | Encoder 共享表征 |
| `obsm` | `cls_emb` | cell-type 分类头 logits |
| `obsm` | `tissue_cls_emb` | tissue 分类头 logits |
| `obsm` | `species_cls_emb` | species 分类头 logits |
| `uns` | `cls_cell_type` | `cls_emb` 列对应的细胞类型顺序 |
| `uns` | `cls_tissue` | `tissue_cls_emb` 列对应的组织顺序 |
| `uns` | `cls_species` | `species_cls_emb` 列对应的物种顺序 |

如果输入 `obs` 中存在相应真实标签，标准推理会计算 Accuracy 和 Macro-F1；分块推理还可以为每块保存 JSON 并汇总到 CSV。

## 文档、Notebook 与发布说明

进一步资料：

- [`docs/API/`](docs/API/)：命令行、`scDataset`、`OntoGRAPH`、`UnicellTrainer`、`unicell_predict` 和指标 API。
- [`docs/PACKAGE ORGANIZATION/`](docs/PACKAGE%20ORGANIZATION/)：模型结构与 HMCN loss。
- [`docs/TUTORIALS/`](docs/TUTORIALS/)：本体引导注释、新细胞类型检测、命名统一、atlas 构建，以及 GeneFormer/scFoundation/scGPT 监督注释教程。
- [`predict.ipynb`](predict.ipynb)：标准推理 Notebook。
- [`predict_by_sample_chunks.ipynb`](predict_by_sample_chunks.ipynb)：大文件分块推理 Notebook。
- [`unicell-train-predicct.ipynb`](unicell-train-predicct.ipynb)：训练与推理 Notebook。

### 数据和制品策略

请勿向 Git 提交患者级数据、H5AD 数据集、checkpoint、预测输出、日志、wheel 或离线依赖包。经批准的模型应通过带版本和校验值的模型/Release 服务发布；数据集应通过具有适当访问控制和许可证的数据仓库发布。

### 第三方代码与许可证状态

`unicell/repo/` 包含与多个单细胞基础模型项目相关的适配或内嵌代码。重新分发前请阅读 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

UniCell 原创贡献采用 [MIT 许可证](LICENSE)，版权署名为 Copyright (c) 2026 UniCell contributors。内嵌的第三方代码和数据仍适用各自的许可证，包括 GPL v3、Apache-2.0、MIT/BSD 和 CC BY 4.0；完整发行版并非仅采用 MIT。来源、版权及尚待补齐的 SCAD/DSBN 授权文档见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [licenses/](licenses/)。代码许可证不会自动覆盖另外下载的模型权重或数据集。
