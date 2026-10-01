# UniCell Fig5 0.2.1 安装后使用说明

本文适用于以下 Python wheel：

~~~text
unicell_fig5-0.2.1-py3-none-any.whl
包名：unicell-fig5
导入名：unicell
版本：0.2.1
~~~

本文档以项目构建生成的以下制品为准：

~~~text
dist/unicell_fig5-0.2.1-py3-none-any.whl
~~~

构建完成后应使用 <code>sha256sum</code> 记录并核对实际 wheel 的校验值。

## 1. 这个包能做什么

UniCell Fig5 0.2.1 面向单细胞表达矩阵，主要提供：

- 细胞类型 Cell Ontology ID 预测；
- 组织预测；
- 物种预测；
- UniCell 表征和各分类头 logits；
- 普通单文件推理；
- 按 <code>adata.obs</code> 中的样本列进行分块推理；
- 使用训练集和验证集重新训练 UniCell 模型。

安装后会出现四个命令：

| 命令 | 用途 |
| --- | --- |
| <code>unicell-predict</code> | 一次读取一个 h5ad 并完成推理 |
| <code>unicell-predict-chunks</code> | 按样本分块推理大 h5ad |
| <code>unicell-train</code> | 使用指定的训练集和验证集训练 |
| <code>unicell-train-backbone</code> | 使用外部 GeneFormer、scGPT 或 expr encoder 进行可配置训练 |

重要：wheel 只包含 Python 代码、本体资源和轻量词表，不包含 UniCell、GeneFormer 或 scGPT 模型权重。推理时必须另外提供完整 UniCell checkpoint 目录；foundation encoder 训练还需要显式提供对应的预训练模型文件。

## 2. 运行环境与模型文件

### 2.1 环境要求

- Linux；
- CPython 3.9，包元数据要求 <code>>=3.9,<3.10</code>；
- PyTorch 2.0.x；
- GPU 推理需要与 PyTorch 构建匹配的 CUDA、驱动和 CUPTI；
- CPU 可以运行，但速度明显慢于 GPU。

wheel 名称中的 <code>py3-none-any</code> 只表示 UniCell 应用代码本身不区分平台。PyTorch、NumPy、SciPy、H5py、Numba 等依赖仍然区分 CPU 架构和 CUDA 版本。

### 2.2 Checkpoint 目录

推荐使用发布包中的：

~~~text
release/common/models/last_version_gml/
~~~

该目录必须至少包含：

~~~text
unicell_v1.best.pth
gene_names.pk
ontoGraph.pk
ontoGraph.graph.gml
celltype_dict.pk
tissue_dict.pk
species_dict.pk
~~~

不要混用来自不同训练结果的这些文件。它们共同定义模型参数、基因顺序、细胞类型、组织、物种和本体图。

只加载可信来源的 checkpoint。<code>.pth</code> 和 <code>.pk</code> 文件使用了可执行的 Python 序列化格式。

### 2.3 Safe list

Safe list 用于限制物种、组织和细胞类型的候选范围。

发布包提供的 <code>safe_list.json</code> 只包含少量 brain、lung、heart 和细胞类型，是格式示例，不是通用预测列表。实际使用时应根据目标组织准备经过审核的 safe list。

只有在输入数据确实符合约束时才使用对应 safe list。错误的 safe list 会强制模型在错误的候选集合中选择。

如果不需要任何约束：

- Python API 中设置 <code>safe_list_path=None</code>；
- 命令行中省略 <code>--safe_list</code>。

## 3. 安装后先做检查

激活安装环境：

~~~bash
source /absolute/path/to/unicell_runtime/bin/activate
~~~

确认 Python 与包版本：

~~~bash
python --version
python -c "from importlib.metadata import version; import unicell; print(version('unicell-fig5')); print(unicell.__version__)"
python -m pip check
~~~

预期 Python 为 3.9.x，两个版本号均为 0.2.1，且 <code>pip check</code> 输出没有依赖冲突。

查看命令帮助：

~~~bash
unicell-predict --help
unicell-predict-chunks --help
unicell-train --help
unicell-train-backbone --help
~~~

如果使用完整发布目录，可以检查安装和 checkpoint：

~~~bash
python /absolute/path/to/release/smoke_test.py \
  --checkpoint-dir /absolute/path/to/release/common/models/last_version_gml
~~~

看到 <code>flash_attn is not installed</code> 通常只是可选加速警告；发布的 <code>expr</code> checkpoint 不依赖 flash-attn。

## 4. 输入 h5ad 要求

### 4.1 必需内容

输入必须是 AnnData h5ad：

- 行是细胞；
- 列是基因；
- 表达矩阵位于 <code>adata.X</code>；
- <code>adata.var_names</code> 是基因标识符；
- <code>adata.obs_names</code> 最好唯一；
- <code>adata.var_names</code> 必须唯一，并尽量与 checkpoint 中的 <code>gene_names.pk</code> 完全一致。

推理时会按 checkpoint 的基因顺序重新排列矩阵：

- 输入中多余的基因会被忽略；
- checkpoint 中缺失的基因会补 0；
- 完全没有匹配基因时会报 <code>No gene names in ref gene</code>。

随发布的 <code>last_version_gml</code> checkpoint 包含 60,695 个输入基因、718 个细胞类型、60 个组织和 2 个物种。这里的基因名可能是复合标识符，不能假设普通 gene symbol 会自动匹配。

### 4.2 标签列是可选的

仅做预测时，不要求输入已有真实标签。

如果希望计算 Accuracy 和 Macro-F1，默认读取：

| 任务 | 默认 obs 列 |
| --- | --- |
| 细胞类型 | <code>cell_type_ontology_term_id</code> |
| 组织 | <code>general_tissue</code> |
| 物种 | <code>organism</code> |

细胞类型真实标签应使用 Cell Ontology ID，例如 <code>CL:0000624</code>。如果列名不同，可用 <code>--cell_type_key</code>、<code>--tissue_key</code> 和 <code>--species_key</code> 指定。

### 4.3 先检查基因匹配数量

~~~python
import pickle
import anndata as ad

input_path = "/absolute/path/to/input.h5ad"
ckpt_dir = "/absolute/path/to/models/last_version_gml"

adata = ad.read_h5ad(input_path, backed="r")
with open(f"{ckpt_dir}/gene_names.pk", "rb") as handle:
    reference_genes = set(map(str, pickle.load(handle)))

input_genes = list(map(str, adata.var_names))
matched = sum(gene in reference_genes for gene in input_genes)

print("input genes:", len(input_genes))
print("checkpoint genes:", len(reference_genes))
print("matched genes:", matched)
print("input var_names unique:", adata.var_names.is_unique)

adata.file.close()
~~~

匹配数为 0 时不能运行。匹配数很少时即使技术上能运行，预测结果也通常不可靠；应先统一基因标识符。

### 4.4 表达值预处理

当前 0.2.1 推理入口会复制输入，并自动执行与训练/验证读取器一致的启发式预处理：

- 对非负矩阵，如果基因数大于 1,000，过滤检测基因少于 200 的细胞；
- 如果表达最大值大于 25，执行 <code>log1p</code>；
- 如果上述最大值看起来是整数 counts，则先执行 <code>normalize_total(target_sum=1e4)</code>。

因此，未经处理的非负整数 counts 可以直接作为输入，不需要提前手动执行相同的
normalize/log1p。已经归一化并 log1p、且最大值不超过 25 的数据不会再次归一化或
log1p，但基因数大于 1,000 时仍会执行细胞过滤。若数据需要不同的预处理策略，应在
调用前自行处理，并确认不会触发上述启发式条件。

## 5. 推荐方式：使用 Python API 推理

Python API 可以关闭 safe list，并便于直接访问预测、embedding 和 logits。本文对应的重建 wheel 已在包内将 <code>level_*</code> 转换为 categorical，因此可以直接写出 h5ad，不再需要额外的 <code>fillna/astype(str)</code> 兼容处理。

将下列内容保存为 <code>run_unicell.py</code>：

~~~python
import os
import anndata as ad
import torch

from unicell.anno_predict import unicell_predict

input_path = "/absolute/path/to/input_log1p.h5ad"
ckpt_dir = "/absolute/path/to/models/last_version_gml"
output_path = "/absolute/path/to/output_annotated.h5ad"

# 不限制候选类别时设为 None。
# 需要约束时改为适合目标组织的 safe list 路径。
safe_list_path = None

device = "cuda" if torch.cuda.is_available() else "cpu"
adata = ad.read_h5ad(input_path)

result = unicell_predict(
    adata=adata,
    filepath=input_path,
    ckpt_dir=ckpt_dir,
    batch_size=512,
    device=device,
    cell_type_key="cell_type_ontology_term_id",
    tissue_key="general_tissue",
    species_key="organism",
    compute_metrics=True,
    safe_list_path=safe_list_path,
)

annotated = result.adata

os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
annotated.write_h5ad(output_path)
print(f"saved: {output_path}")
~~~

运行：

~~~bash
python run_unicell.py
~~~

显存不足时先减小 <code>batch_size</code>，例如 512、256 或 128。<code>batch_size</code> 主要影响模型推理批次，不能消除基因对齐阶段的主机内存占用。

## 6. 使用命令行完成单文件预测

请显式传入输入文件和 checkpoint 路径。输出路径、设备和 safe list 可以按需指定。

~~~bash
unicell-predict \
  --input /absolute/path/to/input_log1p.h5ad \
  --ckpt_dir /absolute/path/to/models/last_version_gml \
  --output /absolute/path/to/output_annotated.h5ad \
  --safe_list /absolute/path/to/safe_list.json \
  --device cuda \
  --batch_size 512 \
  --save_metrics_json
~~~

如需强制使用 CPU：

~~~bash
unicell-predict \
  --input /absolute/path/to/input_log1p.h5ad \
  --ckpt_dir /absolute/path/to/models/last_version_gml \
  --output /absolute/path/to/output_annotated.h5ad \
  --device cpu \
  --batch_size 128
~~~

主要参数：

| 参数 | 说明 |
| --- | --- |
| <code>--input</code> | 输入 h5ad |
| <code>--ckpt_dir</code> | 完整 checkpoint 目录 |
| <code>--output</code> | 输出 h5ad |
| <code>--safe_list</code> | 可选的约束解码 JSON；不需要约束时省略 |
| <code>--device</code> | <code>cuda</code> 或 <code>cpu</code> |
| <code>--batch_size</code> | 推理 batch size |
| <code>--save_metrics_json</code> | 将指标写到与输出同名前缀的 JSON |
| <code>--cell_type_key</code> | 真实细胞类型列名 |
| <code>--tissue_key</code> | 真实组织列名 |
| <code>--species_key</code> | 真实物种列名 |

未指定 <code>--device</code> 时，命令会根据 CUDA 是否可用自动选择 GPU 或 CPU；也可以显式传入 <code>cuda</code> 或 <code>cpu</code>。

### 6.1 层级列保存修复

本文对应的重建 wheel 已修复以下旧制品错误：

~~~text
TypeError: Can't implicitly convert non-string objects to strings
Error raised while writing key 'level_...' of ... /obs
~~~

修复后，所有 <code>level_*</code> 列使用 categorical 保存，较短本体路径仍保留真正的缺失值，不会影响预测标签、指标、embedding 或 logits。若仍出现该错误，请重新构建并使用 <code>--force-reinstall --no-deps</code> 安装当前 wheel，避免 pip 继续使用同版本的旧缓存。

## 7. 大文件按样本分块预测

输入 h5ad 的 <code>obs</code> 中必须有分组列，默认列名为 <code>sample</code>。

~~~bash
unicell-predict-chunks \
  --input /absolute/path/to/large_input_log1p.h5ad \
  --ckpt_dir /absolute/path/to/models/last_version_gml \
  --out_dir /absolute/path/to/prediction_chunks \
  --sample_key sample \
  --safe_list /absolute/path/to/safe_list.json \
  --device cuda \
  --batch_size 512 \
  --compression gzip \
  --compression_opts 4 \
  --skip_done \
  --save_metrics_json \
  --summary_csv /absolute/path/to/prediction_chunks/chunk_summary.csv
~~~

该命令：

1. 以 backed read-only 模式打开原始 h5ad；
2. 按 <code>obs[sample_key]</code> 分组；
3. 每次将一个样本加载到内存；
4. 每个样本输出一个 h5ad 和一个 CSV；
5. 最终生成汇总 CSV；
6. 使用 <code>--skip_done</code> 时跳过已经同时存在 h5ad 和 CSV 的样本。

文件名大致为：

~~~text
pred_000001_SAMPLE.h5ad
pred_000001_SAMPLE.csv
pred_000001_SAMPLE_metrics.json
chunk_summary.csv
~~~

单文件和分块命令在未指定设备时都会根据 CUDA 是否可用自动选择 GPU 或 CPU。

重建 wheel 中的 categorical 修复同时作用于单文件和分块命令；每个样本可以直接保存不同深度的本体路径。

### 7.1 内存估算

基因对齐阶段会先建立一个 <code>细胞数 × 60,695</code> 的 float32 临时数组。仅该数组的近似大小为：

~~~text
细胞数 × 60,695 × 4 bytes
~~~

例如 10,000 个细胞约需 2.26 GiB，尚未计入原始数据、模型、输出 logits 和 Python 开销。

因此：

- 普通命令适合能够整体放入内存的数据；
- 大数据优先使用分块命令；
- 如果单个 sample 仍然很大，应在输入中建立更细的分块列；
- 减小 batch size 只能降低模型批次内存，不能降低上述基因对齐临时数组。

## 8. 输出内容

随发布的 <code>last_version_gml</code> checkpoint 实测输出如下。

### 8.1 obs

| 字段 | 含义 |
| --- | --- |
| <code>predicted_cell_type_ontology_id</code> | 预测 Cell Ontology ID |
| <code>predicted_cell_type</code> | 0.2.1 中同样保存为 ontology ID，并非自然语言名称 |
| <code>predicted_tissue</code> | 预测组织 |
| <code>predicted_species</code> | 预测物种 |
| <code>level_0</code> 等 | categorical 类型；从公共祖先到预测细胞类型的本体路径，不存在的更深层级保留为缺失值 |

### 8.2 obsm

| 字段 | 随附 checkpoint 的形状 | 含义 |
| --- | --- | --- |
| <code>unicell_emb</code> | N × 128 | UniCell 细胞表征 |
| <code>cls_emb</code> | N × 718 | 细胞类型分类 logits |
| <code>tissue_cls_emb</code> | N × 60 | 组织分类 logits |
| <code>species_cls_emb</code> | N × 2 | 物种分类 logits |

这些分类矩阵是 logits，不是已经归一化的概率。

### 8.3 uns

| 字段 | 含义 |
| --- | --- |
| <code>cls_cell_type</code> | <code>cls_emb</code> 各列对应的 Cell Ontology ID |
| <code>cls_tissue</code> | <code>tissue_cls_emb</code> 各列对应的组织 |
| <code>cls_species</code> | <code>species_cls_emb</code> 各列对应的物种 |

读取结果：

~~~python
import anndata as ad

adata = ad.read_h5ad("/absolute/path/to/output_annotated.h5ad")

print(adata.obs[
    [
        "predicted_cell_type_ontology_id",
        "predicted_tissue",
        "predicted_species",
    ]
].head())

print(adata.obsm["unicell_emb"].shape)
print(adata.obsm["cls_emb"].shape)
print(adata.uns["cls_cell_type"][:5])
~~~

### 8.4 保留原始数据

推理输出不是简单地在原文件上增加几列。0.2.1 会把 <code>X</code> 改成 checkpoint 的 60,695 基因顺序，并以 0 填充缺失基因；原有的 <code>var</code> 注释、layers、raw、varm 和 obsp 不会完整保留。

因此应始终保留原始 h5ad。如果希望保持原始表达矩阵，只把预测结果合并回去：

~~~python
import anndata as ad

original = ad.read_h5ad("/absolute/path/to/input_log1p.h5ad")
predicted = ad.read_h5ad("/absolute/path/to/output_annotated.h5ad")

prediction_columns = [
    "predicted_cell_type",
    "predicted_cell_type_ontology_id",
    "predicted_tissue",
    "predicted_species",
]
prediction_columns += [
    name for name in predicted.obs.columns
    if name.startswith("level_")
]

original.obs[prediction_columns] = predicted.obs.loc[
    original.obs_names,
    prediction_columns,
]

for name in [
    "unicell_emb",
    "cls_emb",
    "tissue_cls_emb",
    "species_cls_emb",
]:
    original.obsm[name] = predicted.obsm[name]

for name in ["cls_cell_type", "cls_tissue", "cls_species"]:
    original.uns[name] = predicted.uns[name]

original.write_h5ad("/absolute/path/to/input_with_predictions.h5ad")
~~~

## 9. 指标计算

单文件命令和分块命令会在对应真实标签列存在时计算：

- Accuracy；
- Macro-F1。

不存在真实标签列时会显示 skip，这不会影响预测。

示例：

~~~bash
unicell-predict \
  --input /absolute/path/to/labeled_test.h5ad \
  --ckpt_dir /absolute/path/to/models/last_version_gml \
  --output /absolute/path/to/labeled_test_pred.h5ad \
  --device cuda \
  --batch_size 512 \
  --cell_type_key cell_type_ontology_term_id \
  --tissue_key general_tissue \
  --species_key organism \
  --save_metrics_json
~~~

指标文件路径为：

~~~text
/absolute/path/to/labeled_test_pred_metrics.json
~~~

## 10. 训练

<code>unicell-train</code> 保留为原有 expr 实验入口。它需要两个 h5ad：

~~~bash
unicell-train \
  --train_h5ad /absolute/path/to/train.h5ad \
  --eval_h5ad /absolute/path/to/valid.h5ad
~~~

训练和验证数据都必须包含：

~~~text
obs["cell_type_ontology_term_id"]
obs["general_tissue"]
obs["organism"]
~~~

验证集必须包含训练使用的全部基因，基因标识符要一致。

多 GPU 示例：

~~~bash
torchrun \
  --standalone \
  --nproc_per_node=2 \
  --module train_Unicell \
  --train_h5ad /absolute/path/to/train.h5ad \
  --eval_h5ad /absolute/path/to/valid.h5ad
~~~

原有 <code>unicell-train</code> 是固定 expr 实验脚本，只有训练集和验证集路径可通过命令行修改。以下配置写死在该旧入口内：

- checkpoint 输出目录为当前工作目录下的 <code>models/checkpoints</code>；
- batch size 为 128；
- learning rate 为 1e-3；
- epoch 数为 50；
- 标签列名固定为上述三列。

该旧入口会使用训练集建立 tissue/species 类别编码，并将验证集映射到同一编码。验证集若包含训练集中不存在的 tissue/species 标签，程序会明确报错。新的可配置任务请使用下面的 <code>unicell-train-backbone</code>。

### 10.1 使用 GeneFormer encoder

GeneFormer 的 <code>--llm_model_file</code> 必须是包含 <code>config.json</code> 和本地权重的 Hugging Face 模型目录。UniCell 默认使用 wheel 内置的 GeneFormer 词表：

~~~bash
unicell-train-backbone \
  --input_type GeneFormer \
  --train_h5ad /absolute/path/to/train.h5ad \
  --eval_h5ad /absolute/path/to/valid.h5ad \
  --llm_model_file /absolute/path/to/Geneformer-V1-10M \
  --ckpt_dir /absolute/path/to/output_geneformer \
  --validate_only
~~~

### 10.2 使用 scGPT encoder

scGPT 的模型、词表和 <code>args.json</code> 必须来自同一个预训练发布：

~~~bash
unicell-train-backbone \
  --input_type scGPT \
  --train_h5ad /absolute/path/to/train.h5ad \
  --eval_h5ad /absolute/path/to/valid.h5ad \
  --llm_model_file /absolute/path/to/scgpt/best_model.pt \
  --llm_vocab_file /absolute/path/to/scgpt/vocab.json \
  --llm_args_file /absolute/path/to/scgpt/args.json \
  --ckpt_dir /absolute/path/to/output_scgpt \
  --validate_only
~~~

两条命令都建议先保留 <code>--validate_only</code> 完成 h5ad、词表和模型结构检查；确认通过后删除该参数开始训练。多 GPU 可使用：

~~~bash
torchrun --standalone --nproc_per_node=2 \
  -m unicell.cli.train_backbone \
  ...上述参数...
~~~

### 10.3 训练后移动和推理 checkpoint

GeneFormer/scGPT 训练结果会在普通 UniCell 七件套之外增加：

~~~text
output_checkpoint/
├── unicell_v1.best.pth
├── gene_names.pk / ontology 与标签字典文件
├── run_config.json
└── backbone/
    ├── manifest.json
    ├── config.json + gene_vocab.json    # GeneFormer
    └── args.json + vocab.json           # scGPT
~~~

完整训练权重已经位于 <code>unicell_v1.best.pth</code>。整体移动 <code>output_checkpoint</code> 后，安装 wheel 的环境只需指定新目录即可推理，不再需要原始 foundation checkpoint 路径：

~~~bash
unicell-predict \
  --input /absolute/path/to/test.h5ad \
  --ckpt_dir /absolute/path/to/moved_output_checkpoint \
  --output /absolute/path/to/test_annotated.h5ad \
  --device cuda
~~~

旧版 GeneFormer/scGPT checkpoint 如果 metadata 中仍是另一台机器的绝对路径，可分别使用 <code>--backbone_model</code>、<code>--backbone_vocab</code> 和 <code>--backbone_args</code> 覆盖。例如：

~~~bash
unicell-predict \
  --input /absolute/path/to/test.h5ad \
  --ckpt_dir /absolute/path/to/legacy_scgpt_checkpoint \
  --backbone_model /absolute/path/to/scgpt/best_model.pt \
  --backbone_vocab /absolute/path/to/scgpt/vocab.json \
  --backbone_args /absolute/path/to/scgpt/args.json \
  --output /absolute/path/to/test_annotated.h5ad
~~~

旧 GeneFormer checkpoint 只需将 <code>--backbone_model</code> 指向本地 Hugging Face 模型目录；需要时可用 <code>--backbone_vocab</code> 覆盖词表。<code>--init_backbone_from_config</code> 是迁移用高级选项，只应在完整 UniCell state dict 与重建配置兼容时启用。相同参数也适用于 <code>unicell-predict-chunks</code>；<code>input_type=expr</code> checkpoint 不进入 foundation encoder 分支。

## 11. 常见问题

### Python 版本不兼容

~~~text
ERROR: Package 'unicell-fig5' requires a different Python
~~~

使用 Python 3.9 新建环境，不要使用 3.10、3.11、3.12 或 3.13。

### 找不到输入、模型或 safe list

单文件和分块命令都要求显式提供 <code>--input</code> 与 <code>--ckpt_dir</code>；分块命令还要求 <code>--out_dir</code>。输出路径和 <code>--safe_list</code> 的可选行为以 <code>--help</code> 为准。

### 找不到匹配基因

~~~text
ValueError: No gene names in ref gene
~~~

检查 <code>adata.var_names</code> 与 <code>gene_names.pk</code> 使用的是 gene symbol、Ensembl ID 还是复合标识符，并统一命名。

### CUDA 不可用或显存不足

- 先运行 <code>python -c "import torch; print(torch.cuda.is_available())"</code>；
- 无 GPU 时给单文件命令加 <code>--device cpu</code>；
- 显存不足时减小 batch size；
- 主机内存不足时使用更小的分块。

### libstdc++、CXXABI 或 libcupti 错误

这通常是集群模块、Conda 环境与 CUDA 动态库路径冲突。优先在干净 shell 中激活环境，并让环境自身的库排在系统旧库之前：

~~~bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/path/to/matching/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
~~~

CUDA 路径必须与安装的 PyTorch 构建匹配。不要把 CUDA 12 的库用于要求 CUDA 11.6 或 11.7 的 PyTorch 2.0 构建。

### flash-attn 警告

~~~text
UserWarning: flash_attn is not installed
~~~

对于随附的 <code>input_type=expr</code> checkpoint 可以忽略。

### 仍然出现 level 列保存错误

本文对应的 wheel 已包含 categorical 修复。构建后运行 <code>sha256sum unicell_fig5-0.2.1-py3-none-any.whl</code> 保存实际校验值；请升级到 0.2.1 wheel，并按第 3 节的命令核对已安装版本。如果需要替换本地重新构建的同版本 wheel，可使用 <code>--force-reinstall --no-deps</code>，防止 pip 沿用旧制品。

## 12. 卸载

~~~bash
python -m pip uninstall unicell-fig5
~~~

checkpoint、safe list、输入和输出数据不属于 pip 安装内容，需要按实际存放位置单独管理。
