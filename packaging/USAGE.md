# UniCell Fig5 0.2.1 Post-installation Usage Guide

This guide applies to the following Python wheel:

~~~text
unicell_fig5-0.2.1-py3-none-any.whl
Package name: unicell-fig5
Import name: unicell
Version: 0.2.1
~~~

This document refers to the following artifact produced by the project build:

~~~text
dist/unicell_fig5-0.2.1-py3-none-any.whl
~~~

After building, use <code>sha256sum</code> to record and verify the checksum of the actual wheel.

## 1. What This Package Provides

UniCell Fig5 0.2.1 works with single-cell expression matrices and primarily provides:

- Cell type prediction as Cell Ontology IDs;
- Tissue prediction;
- Species prediction;
- UniCell representations and logits from each classification head;
- Standard single-file inference;
- Chunked inference by a sample column in <code>adata.obs</code>;
- Retraining of the UniCell model using training and validation datasets.

Installation provides four commands:

| Command | Purpose |
| --- | --- |
| <code>unicell-predict</code> | Read one h5ad file at a time and run inference |
| <code>unicell-predict-chunks</code> | Run inference on a large h5ad file in chunks grouped by sample |
| <code>unicell-train</code> | Train using the specified training and validation datasets |
| <code>unicell-train-backbone</code> | Run configurable training with an external GeneFormer or scGPT encoder, or an expr encoder |

Important: the wheel contains only Python code, ontology resources, and lightweight vocabularies. It does not contain UniCell, GeneFormer, or scGPT model weights. Inference requires a complete UniCell checkpoint directory supplied separately; training with a foundation encoder also requires explicitly supplied pretrained model files for that encoder.

## 2. Runtime Environment and Model Files

### 2.1 Environment Requirements

- Linux;
- CPython 3.9; the package metadata requires <code>>=3.9,<3.10</code>;
- PyTorch 2.0.x;
- GPU inference requires CUDA, drivers, and CUPTI compatible with the PyTorch build;
- CPU execution is supported, but is significantly slower than GPU execution.

The <code>py3-none-any</code> tag in the wheel filename only indicates that the UniCell application code itself is platform-independent. Dependencies such as PyTorch, NumPy, SciPy, H5py, and Numba remain specific to the CPU architecture and CUDA version.

### 2.2 Checkpoint Directory

The recommended directory in the release package is:

~~~text
release/common/models/last_version_gml/
~~~

This directory must contain at least:

~~~text
unicell_v1.best.pth
gene_names.pk
ontoGraph.pk
ontoGraph.graph.gml
celltype_dict.pk
tissue_dict.pk
species_dict.pk
~~~

Do not mix these files from different training runs. Together, they define the model parameters, gene order, cell types, tissues, species, and ontology graph.

Only load checkpoints from trusted sources. The <code>.pth</code> and <code>.pk</code> files use Python serialization formats that can execute code.

### 2.3 Safe List

A safe list restricts the candidate species, tissues, and cell types.

The <code>safe_list.json</code> supplied in the release contains only a small selection of brain, lung, heart, and cell type entries. It is a format example, not a general-purpose prediction list. For actual use, prepare a reviewed safe list appropriate for the target tissue.

Use a safe list only when the input data truly meet its constraints. An incorrect safe list forces the model to choose from an incorrect set of candidates.

If no constraints are needed:

- Set <code>safe_list_path=None</code> in the Python API;
- Omit <code>--safe_list</code> on the command line.

## 3. Initial Checks After Installation

Activate the installation environment:

~~~bash
source /absolute/path/to/unicell_runtime/bin/activate
~~~

Check the Python and package versions:

~~~bash
python --version
python -c "from importlib.metadata import version; import unicell; print(version('unicell-fig5')); print(unicell.__version__)"
python -m pip check
~~~

Python should be 3.9.x, both package version numbers should be 0.2.1, and <code>pip check</code> should report no dependency conflicts.

View command help:

~~~bash
unicell-predict --help
unicell-predict-chunks --help
unicell-train --help
unicell-train-backbone --help
~~~

If you are using the complete release directory, you can check the installation and checkpoint:

~~~bash
python /absolute/path/to/release/smoke_test.py \
  --checkpoint-dir /absolute/path/to/release/common/models/last_version_gml
~~~

The message <code>flash_attn is not installed</code> is usually only a warning about optional acceleration; the released <code>expr</code> checkpoint does not depend on flash-attn.

## 4. Input h5ad Requirements

### 4.1 Required Contents

The input must be an AnnData h5ad file:

- Rows represent cells;
- Columns represent genes;
- The expression matrix is stored in <code>adata.X</code>;
- <code>adata.var_names</code> contains gene identifiers;
- <code>adata.obs_names</code> should preferably be unique;
- <code>adata.var_names</code> must be unique and should match <code>gene_names.pk</code> in the checkpoint as closely as possible.

During inference, the matrix is reordered to match the checkpoint's gene order:

- Extra genes in the input are ignored;
- Genes required by the checkpoint but missing from the input are filled with 0;
- If no genes match, the error <code>No gene names in ref gene</code> is raised.

The released <code>last_version_gml</code> checkpoint contains 60,695 input genes, 718 cell types, 60 tissues, and 2 species. Its gene names may be composite identifiers; do not assume that standard gene symbols will match automatically.

### 4.2 Label Columns Are Optional

Ground-truth labels are not required for prediction alone.

To calculate Accuracy and Macro-F1, the following columns are read by default:

| Task | Default obs Column |
| --- | --- |
| Cell type | <code>cell_type_ontology_term_id</code> |
| Tissue | <code>general_tissue</code> |
| Species | <code>organism</code> |

Ground-truth cell type labels should use Cell Ontology IDs, such as <code>CL:0000624</code>. If your column names differ, specify them using <code>--cell_type_key</code>, <code>--tissue_key</code>, and <code>--species_key</code>.

### 4.3 Check the Number of Matching Genes First

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

Inference cannot run when there are 0 matching genes. When very few genes match, predictions are generally unreliable even if inference can technically run; standardize the gene identifiers first.

### 4.4 Expression Preprocessing

The current 0.2.1 inference entry points copy the input and automatically apply the same heuristic preprocessing as the training/validation readers:

- For nonnegative matrices with more than 1,000 genes, filter out cells with fewer than 200 detected genes;
- If the maximum expression value exceeds 25, apply <code>log1p</code>;
- If that maximum value appears to be an integer count, first apply <code>normalize_total(target_sum=1e4)</code>.

Therefore, unprocessed nonnegative integer counts can be used directly as input, without manually applying the same
normalization/log1p steps beforehand. Data that have already been normalized and log1p-transformed, with a maximum value no greater than 25, will not be normalized or
log1p-transformed again, but cell filtering still applies when there are more than 1,000 genes. If your data require a different preprocessing strategy, process them yourself
before calling the API and confirm that they will not trigger the heuristic conditions above.

## 5. Recommended Approach: Inference with the Python API

The Python API allows you to disable the safe list and easily access predictions, embeddings, and logits. The rebuilt wheel covered by this guide converts <code>level_*</code> columns to categorical within the package, so h5ad files can be written directly without additional <code>fillna/astype(str)</code> compatibility handling.

Save the following as <code>run_unicell.py</code>:

~~~python
import os
import anndata as ad
import torch

from unicell.anno_predict import unicell_predict

input_path = "/absolute/path/to/input_log1p.h5ad"
ckpt_dir = "/absolute/path/to/models/last_version_gml"
output_path = "/absolute/path/to/output_annotated.h5ad"

# Set to None to leave candidate classes unrestricted.
# To apply constraints, use the path to a safe list appropriate for the target tissue.
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

Run:

~~~bash
python run_unicell.py
~~~

If GPU memory is insufficient, first reduce <code>batch_size</code>, for example to 512, 256, or 128. <code>batch_size</code> primarily controls model inference batches and cannot eliminate host memory usage during gene alignment.

## 6. Single-File Prediction from the Command Line

Explicitly provide the input file and checkpoint paths. The output path, device, and safe list can be specified as needed.

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

To force CPU execution:

~~~bash
unicell-predict \
  --input /absolute/path/to/input_log1p.h5ad \
  --ckpt_dir /absolute/path/to/models/last_version_gml \
  --output /absolute/path/to/output_annotated.h5ad \
  --device cpu \
  --batch_size 128
~~~

Main arguments:

| Argument | Description |
| --- | --- |
| <code>--input</code> | Input h5ad file |
| <code>--ckpt_dir</code> | Complete checkpoint directory |
| <code>--output</code> | Output h5ad file |
| <code>--safe_list</code> | Optional JSON for constrained decoding; omit if no constraints are needed |
| <code>--device</code> | <code>cuda</code> or <code>cpu</code> |
| <code>--batch_size</code> | Inference batch size |
| <code>--save_metrics_json</code> | Write metrics to a JSON file with the same filename prefix as the output |
| <code>--cell_type_key</code> | Column name for ground-truth cell types |
| <code>--tissue_key</code> | Column name for ground-truth tissues |
| <code>--species_key</code> | Column name for ground-truth species |

When <code>--device</code> is omitted, the command automatically chooses GPU or CPU based on CUDA availability; you can also explicitly specify <code>cuda</code> or <code>cpu</code>.

### 6.1 Fix for Saving Hierarchy Columns

The rebuilt wheel covered by this guide fixes the following error in older artifacts:

~~~text
TypeError: Can't implicitly convert non-string objects to strings
Error raised while writing key 'level_...' of ... /obs
~~~

With this fix, all <code>level_*</code> columns are saved as categorical, and shorter ontology paths retain genuine missing values. Prediction labels, metrics, embeddings, and logits are unaffected. If this error persists, rebuild and install the current wheel using <code>--force-reinstall --no-deps</code> to prevent pip from reusing an older cached artifact with the same version number.

## 7. Chunked Prediction of Large Files by Sample

The input h5ad must have a grouping column in <code>obs</code>; the default column name is <code>sample</code>.

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

This command:

1. Opens the original h5ad in backed read-only mode;
2. Groups cells by <code>obs[sample_key]</code>;
3. Loads one sample into memory at a time;
4. Writes one h5ad and one CSV per sample;
5. Generates a summary CSV at the end;
6. With <code>--skip_done</code>, skips samples for which both h5ad and CSV files already exist.

Filenames follow this general pattern:

~~~text
pred_000001_SAMPLE.h5ad
pred_000001_SAMPLE.csv
pred_000001_SAMPLE_metrics.json
chunk_summary.csv
~~~

Both the single-file and chunked commands automatically choose GPU or CPU based on CUDA availability when no device is specified.

The categorical fix in the rebuilt wheel applies to both single-file and chunked commands; ontology paths of different depths can be saved directly for each sample.

### 7.1 Memory Estimate

Gene alignment first creates a temporary float32 array of shape <code>number of cells × 60,695</code>. The approximate size of this array alone is:

~~~text
number of cells × 60,695 × 4 bytes
~~~

For example, 10,000 cells require approximately 2.26 GiB, excluding the original data, model, output logits, and Python overhead.

Therefore:

- The standard command is suitable for data that fit entirely in memory;
- Prefer the chunked command for large datasets;
- If a single sample is still too large, create a more fine-grained chunking column in the input;
- Reducing batch size only reduces model batch memory; it does not reduce the temporary gene-alignment array described above.

## 8. Output Contents

The outputs observed with the released <code>last_version_gml</code> checkpoint are described below.

### 8.1 obs

| Field | Meaning |
| --- | --- |
| <code>predicted_cell_type_ontology_id</code> | Predicted Cell Ontology ID |
| <code>predicted_cell_type</code> | Also stored as an ontology ID in 0.2.1, rather than a natural-language name |
| <code>predicted_tissue</code> | Predicted tissue |
| <code>predicted_species</code> | Predicted species |
| <code>level_0</code>, etc. | Categorical columns representing the ontology path from the common ancestor to the predicted cell type; nonexistent deeper levels remain missing values |

### 8.2 obsm

| Field | Shape with the Supplied Checkpoint | Meaning |
| --- | --- | --- |
| <code>unicell_emb</code> | N × 128 | UniCell cell representations |
| <code>cls_emb</code> | N × 718 | Cell type classification logits |
| <code>tissue_cls_emb</code> | N × 60 | Tissue classification logits |
| <code>species_cls_emb</code> | N × 2 | Species classification logits |

These classification matrices contain logits, not normalized probabilities.

### 8.3 uns

| Field | Meaning |
| --- | --- |
| <code>cls_cell_type</code> | Cell Ontology IDs corresponding to the columns of <code>cls_emb</code> |
| <code>cls_tissue</code> | Tissues corresponding to the columns of <code>tissue_cls_emb</code> |
| <code>cls_species</code> | Species corresponding to the columns of <code>species_cls_emb</code> |

Read the results:

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

### 8.4 Preserve the Original Data

The inference output does not simply add a few columns to the original file. Version 0.2.1 changes <code>X</code> to match the checkpoint's order of 60,695 genes and fills missing genes with 0; the original <code>var</code> annotations, layers, raw, varm, and obsp are not fully preserved.

Always retain the original h5ad file. To keep the original expression matrix, merge only the prediction results back into it:

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

## 9. Metrics

The single-file and chunked commands calculate the following when the corresponding ground-truth label columns are present:

- Accuracy;
- Macro-F1.

If a ground-truth label column is absent, a skip message is displayed; prediction is unaffected.

Example:

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

The metrics file path is:

~~~text
/absolute/path/to/labeled_test_pred_metrics.json
~~~

## 10. Training

<code>unicell-train</code> is retained as the original expr experiment entry point. It requires two h5ad files:

~~~bash
unicell-train \
  --train_h5ad /absolute/path/to/train.h5ad \
  --eval_h5ad /absolute/path/to/valid.h5ad
~~~

Both the training and validation datasets must contain:

~~~text
obs["cell_type_ontology_term_id"]
obs["general_tissue"]
obs["organism"]
~~~

The validation dataset must contain all genes used for training, with matching gene identifiers.

Multi-GPU example:

~~~bash
torchrun \
  --standalone \
  --nproc_per_node=2 \
  --module train_Unicell \
  --train_h5ad /absolute/path/to/train.h5ad \
  --eval_h5ad /absolute/path/to/valid.h5ad
~~~

The original <code>unicell-train</code> is a fixed expr experiment script; only the training and validation dataset paths can be changed through the command line. The following settings are hard-coded in this legacy entry point:

- The checkpoint output directory is <code>models/checkpoints</code> under the current working directory;
- Batch size is 128;
- Learning rate is 1e-3;
- The number of epochs is 50;
- The label column names are fixed to the three columns listed above.

This legacy entry point creates tissue/species class encodings from the training dataset and maps the validation dataset to the same encodings. If the validation dataset contains tissue/species labels absent from the training dataset, the program reports an explicit error. For new tasks that require configurable training, use <code>unicell-train-backbone</code> as described below.

### 10.1 Using a GeneFormer Encoder

For GeneFormer, <code>--llm_model_file</code> must point to a Hugging Face model directory containing <code>config.json</code> and local weights. UniCell uses the GeneFormer vocabulary bundled in the wheel by default:

~~~bash
unicell-train-backbone \
  --input_type GeneFormer \
  --train_h5ad /absolute/path/to/train.h5ad \
  --eval_h5ad /absolute/path/to/valid.h5ad \
  --llm_model_file /absolute/path/to/Geneformer-V1-10M \
  --ckpt_dir /absolute/path/to/output_geneformer \
  --validate_only
~~~

### 10.2 Using an scGPT Encoder

The scGPT model, vocabulary, and <code>args.json</code> must come from the same pretrained release:

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

For both commands, first keep <code>--validate_only</code> to check the h5ad files, vocabulary, and model structure; after the checks pass, remove this argument to start training. For multiple GPUs, use:

~~~bash
torchrun --standalone --nproc_per_node=2 \
  -m unicell.cli.train_backbone \
  ...arguments above...
~~~

### 10.3 Moving Checkpoints and Running Inference After Training

GeneFormer/scGPT training produces the following additional files alongside the usual seven UniCell checkpoint files:

~~~text
output_checkpoint/
├── unicell_v1.best.pth
├── gene_names.pk / ontology and label dictionary files
├── run_config.json
└── backbone/
    ├── manifest.json
    ├── config.json + gene_vocab.json    # GeneFormer
    └── args.json + vocab.json           # scGPT
~~~

The complete trained weights are already stored in <code>unicell_v1.best.pth</code>. After moving the entire <code>output_checkpoint</code> directory, an environment with the wheel installed can run inference by specifying only the new directory; the original foundation checkpoint path is no longer needed:

~~~bash
unicell-predict \
  --input /absolute/path/to/test.h5ad \
  --ckpt_dir /absolute/path/to/moved_output_checkpoint \
  --output /absolute/path/to/test_annotated.h5ad \
  --device cuda
~~~

If the metadata of a legacy GeneFormer/scGPT checkpoint still contain absolute paths from another machine, override them with <code>--backbone_model</code>, <code>--backbone_vocab</code>, and <code>--backbone_args</code>, respectively. For example:

~~~bash
unicell-predict \
  --input /absolute/path/to/test.h5ad \
  --ckpt_dir /absolute/path/to/legacy_scgpt_checkpoint \
  --backbone_model /absolute/path/to/scgpt/best_model.pt \
  --backbone_vocab /absolute/path/to/scgpt/vocab.json \
  --backbone_args /absolute/path/to/scgpt/args.json \
  --output /absolute/path/to/test_annotated.h5ad
~~~

For legacy GeneFormer checkpoints, point <code>--backbone_model</code> to the local Hugging Face model directory; use <code>--backbone_vocab</code> to override the vocabulary if needed. <code>--init_backbone_from_config</code> is an advanced migration option and should only be enabled when the complete UniCell state dict is compatible with the reconstructed configuration. The same arguments apply to <code>unicell-predict-chunks</code>; <code>input_type=expr</code> checkpoints do not enter the foundation encoder branch.

## 11. Troubleshooting

### Incompatible Python Version

~~~text
ERROR: Package 'unicell-fig5' requires a different Python
~~~

Create a new environment using Python 3.9; do not use 3.10, 3.11, 3.12, or 3.13.

### Input, Model, or Safe List Not Found

Both single-file and chunked commands require explicitly supplied <code>--input</code> and <code>--ckpt_dir</code>; the chunked command also requires <code>--out_dir</code>. Consult <code>--help</code> for the optional behavior of output paths and <code>--safe_list</code>.

### No Matching Genes Found

~~~text
ValueError: No gene names in ref gene
~~~

Check whether <code>adata.var_names</code> and <code>gene_names.pk</code> use gene symbols, Ensembl IDs, or composite identifiers, and make the naming consistent.

### CUDA Unavailable or Insufficient GPU Memory

- First run <code>python -c "import torch; print(torch.cuda.is_available())"</code>;
- Without a GPU, add <code>--device cpu</code> to the single-file command;
- Reduce batch size if GPU memory is insufficient;
- Use smaller chunks if host memory is insufficient.

### libstdc++, CXXABI, or libcupti Errors

These usually result from conflicts between cluster modules, Conda environments, and CUDA shared-library paths. Prefer activating the environment in a clean shell and placing the environment's own libraries before older system libraries:

~~~bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/path/to/matching/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
~~~

The CUDA path must match the installed PyTorch build. Do not use CUDA 12 libraries with PyTorch 2.0 builds that require CUDA 11.6 or 11.7.

### flash-attn Warning

~~~text
UserWarning: flash_attn is not installed
~~~

This can be ignored for the supplied <code>input_type=expr</code> checkpoint.

### Errors When Saving level Columns Persist

The wheel covered by this guide already includes the categorical fix. After building, run <code>sha256sum unicell_fig5-0.2.1-py3-none-any.whl</code> to record the actual checksum. Upgrade to the 0.2.1 wheel and verify the installed version using the command in Section 3. If replacing a locally rebuilt wheel with the same version, use <code>--force-reinstall --no-deps</code> to prevent pip from reusing an older artifact.

## 12. Uninstallation

~~~bash
python -m pip uninstall unicell-fig5
~~~

Checkpoints, safe lists, input data, and output data are not installed by pip and must be managed separately at their actual storage locations.
