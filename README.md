# UniCell

**English** | [简体中文](README_ZH.md)

UniCell is an ontology-guided annotation framework for single-cell transcriptomic data. The current implementation takes H5AD files as input, predicts cell type, tissue, and species simultaneously, and can optionally use a hierarchical safe list at inference time to restrict the final candidate labels in the order `species -> tissue -> cell_type`.

This repository contains training and inference code, a Python package, notebooks, API/tutorial documentation, and helper scripts for wheel and offline releases. Datasets, training checkpoints, inference results, and build artifacts are not committed directly to Git.

> [!IMPORTANT]
> The Python **distribution name** in the current `pyproject.toml` is `unicell-fig5`, while the Python **import name** is `unicell`. After the project formally claims and publishes the `unicell` distribution name, the intended installation command will be `pip install unicell`. Until then, use a local wheel, `pip install .`, or `pip install unicell-fig5` after it has been published to a package index. Do not confuse the import name with the distribution name.

## Table of Contents

- [Feature Overview](#feature-overview)
- [Project Structure](#project-structure)
- [Environment and Installation](#environment-and-installation)
- [Checkpoint Files](#checkpoint-files)
- [H5AD Input Requirements](#h5ad-input-requirements)
- [Safe List: Optional Candidate-Label Constraints](#safe-list-optional-candidate-label-constraints)
- [Usage After Installing the Python Package](#usage-after-installing-the-python-package)
- [Running the Repository's Python Files Directly](#running-the-repositorys-python-files-directly)
- [Inference Output](#inference-outputs)
- [Documentation, Notebooks, and Release Notes](#documentation-notebooks-and-release-notes)

## Feature Overview

- Predicts `cell_type`, `tissue`, and `species` simultaneously.
- Models label hierarchies through the Cell Ontology graph and writes ontology hierarchy paths to the results.
- Supports standard expression-matrix input and configurable UniCell training/inference with GeneFormer or scGPT encoders.
- Supports inference on a single H5AD file, sample-wise chunked inference on large files, single-GPU training, and `torchrun` DDP training.
- Supports constraints on species only, tissue only, or cell type only, as well as a hierarchical species–tissue–cell type safe list.
- Writes predicted labels, the shared representation, logits from all three classification heads, and class ordering back to AnnData.

## Project Structure

The structure below shows only the main usage-related files in the Git repository. Local build artifacts such as `build/`, `dist/`, `*.egg-info/`, and `__pycache__/` are not part of the source tree.

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

### Main Python Files in the Repository Root

| File | Primary purpose | Entry point after installing the wheel |
|---|---|---|
| `train_Unicell.py` | Reads fixed training and validation datasets; establishes the training gene order and ontology; validates and aligns the validation set; initializes single-GPU or DDP training; saves the best checkpoint | `unicell-train` |
| `predict.py` | Runs inference on a single H5AD file; writes the three prediction types, embedding, logits, and ontology hierarchy; computes Accuracy/Macro-F1 when ground-truth labels are present | `unicell-predict` |
| `predict_by_sample_chunks.py` | Opens a large H5AD file in backed mode, loads each sample into memory according to `obs[sample_key]`, and outputs H5AD/CSV files; supports resuming by skipping completed outputs, compression, per-chunk metrics, and a summary table | `unicell-predict-chunks` |
| `train_epoch_unicell.py` | Reads `seed_i/train.h5ad` in each epoch while keeping the reference genes, ontology, class dictionaries, and validation set fixed; intended for a specific experiment that swaps data each epoch | Source execution only |
| `run_umap.py` | First runs UniCell inference without a safe list, then computes UMAPs for the shared representation and the outputs of the three classification heads and exports images/H5AD files | Source execution only |

### Core `unicell/` Package

| File or directory | Primary purpose |
|---|---|
| `anno_predict.py` | Loads a checkpoint, aligns input to the training gene order, constructs HMCN, performs batched inference, applies optional safe-list decoding, and writes the results back to AnnData |
| `scDataset.py` | Reads or wraps AnnData; builds the Cell Ontology subgraph required for training; handles legacy ontology IDs, cell-type indices, and tissue/species encoding |
| `dataset.py` | PyTorch Dataset and data-transformation layer; supports expression matrices, scFoundation, GeneFormer, scGPT, and LMDB data |
| `hmcn.py` | Defines the encoder, HMCN global/local branches, and the cell type, tissue, and species classification heads |
| `loss.py` | Defines HMCN global/local hierarchical losses, Focal Loss, and an optional MMD batch loss |
| `trainer.py` | Assembles the model, DataLoader, losses, and optimizer; performs training/validation; saves checkpoints according to the best F1 score; records training time |
| `ontoGRAPH.py` | Imports, prunes, and queries the Cell Ontology; generates hierarchical mappings and supervision matrices; serializes ontology artifacts for checkpoints |
| `evaluate.py` | Computes Accuracy, Macro-F1, and Micro-F1 from label columns in AnnData |
| `utils/` | Metrics, ontology, LMDB, logging, configuration, and compatibility helper functions |
| `repo/` | Adapter or bundled third-party code for GeneFormer, scGPT, scFoundation, and scBERT; read `THIRD_PARTY_NOTICES.md` before redistribution |
| `cli/train_backbone.py` | Canonical packaged CLI for validated GeneFormer/scGPT/expr training and portable checkpoint creation |

Other directories:

- `docs/`: API, model-architecture, and tutorial documentation.
- `packaging/`: Scripts for wheel builds, multi-architecture releases, offline wheelhouses, installation, and smoke tests.
- `demo/`: Cell Ontology ID reference lists for human and mouse atlases; does not contain directly runnable H5AD files or checkpoints.
- The three notebooks in the repository root: interactive examples for standard inference, chunked inference, and training/inference.

## Environment and Installation

### Basic Requirements

- Linux
- Python `>=3.9,<3.10`; that is, the current version requires Python 3.9
- PyTorch 2.0.x
- GPU inference/training requires a CUDA environment compatible with PyTorch. CPU execution is also supported but is generally considerably slower.

For dependencies and x86_64/aarch64 offline release procedures, see [`packaging/README.md`](packaging/README.md) and [`packaging/USAGE.md`](packaging/USAGE.md).

### Installing from a Package Index

After the project is formally published under the `unicell` distribution name, users are expected to download and install the official wheel with:

```bash
python -m pip install unicell
```

However, **the current repository metadata does not yet use that distribution name**. The distribution name in `pyproject.toml` is currently `unicell-fig5`, so the correct package-index installation command at present is:

```bash
# Valid only after unicell-fig5 has been published to the PyPI/private index in use
python -m pip install unicell-fig5
```

To formally enable `pip install unicell`, the publisher must first confirm that this project owns that name on the package index, change `[project].name` to `unicell`, rebuild the wheel, and publish the new wheel to the relevant index. Editing the README alone cannot change the wheel's installation name.

### Installing a Local Wheel

```bash
python -m pip install /path/to/unicell_fig5-0.2.1-py3-none-any.whl
```

The wheel contains the Python package, four console commands, ontology resources,
and lightweight tokenizer vocabularies. It does **not** contain a UniCell
checkpoint or GeneFormer/scGPT pretrained weights, nor does it contain
`safe_list.json`, `run_umap.py`, or `train_epoch_unicell.py`. Supply the local
foundation-model files when training; a complete portable checkpoint created by
`unicell-train-backbone` no longer needs those original paths for inference.

### Installing from Source

```bash
git clone https://github.com/luyi98/unicell.git
cd unicell

python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For an editable development installation:

```bash
python -m pip install -e .
```

Verify the installation:

```bash
python -c "import unicell; print(unicell.__version__)"
unicell-predict --help
unicell-predict-chunks --help
unicell-train --help
unicell-train-backbone --help
```

## Checkpoint Files

Model files are not committed to Git. The versioned v0.2.1 release provides the
expression checkpoint and its integrity metadata:

- [Download the UniCell expression checkpoint](https://github.com/luyi98/unicell/releases/download/v0.2.0/unicell-last_version_gml-v1.tar.gz)
- [Download the checkpoint manifest](https://github.com/luyi98/unicell/releases/download/v0.2.1/checkpoint-manifest.json)
- [Download the SHA256 checksum file](https://github.com/luyi98/unicell/releases/download/v0.2.1/SHA256SUMS)

The checkpoint archive is reused directly from v0.2.0 instead of being uploaded
again. Its manifest is unchanged; the original training and source provenance
is retained in that manifest. See the
[v0.2.1 release notes](docs/RELEASE_NOTES_0.2.1.md) for the code changes and
validation limits.

### Base Checkpoint Files

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

| File | Purpose |
|---|---|
| `unicell_v1.best.pth` | PyTorch checkpoint containing the `model_state_dict` and model-structure metadata, such as input type, input/output dimensions, global/local layers, dropout, and external foundation-model paths |
| `gene_names.pk` | Ordered list of training genes; inference reorders the input to match this sequence, ignores extra genes, and fills with zeros any genes that exist in the checkpoint but are missing from the input |
| `ontoGraph.pk` | Serialized `OntoGRAPH` object that stores the vocabulary, hierarchy, hierarchical arrays, and class indices; the object's actual graph is cleared when it is saved |
| `ontoGraph.graph.gml` | The actual ontology subgraph paired with `ontoGraph.pk`; it is reattached to the object during loading |
| `celltype_dict.pk` | Mapping from cell-type labels to indices in the flat cell-type classification head; current labels are usually Cell Ontology IDs |
| `tissue_dict.pk` | Mapping from tissue names to indices in the tissue classification head |
| `species_dict.pk` | Mapping from species names to indices in the species classification head |

Notes:

- All seven files must come from the same compatible training release; files from different versions cannot be mixed. The classification-head dimensions, dictionaries, gene order, and ontology must all agree.
- `unicell/graph.gml` in the package is used to construct the ontology during training and is not the same as `ontoGraph.graph.gml` in a checkpoint.
- `train_time_summary.json` and `timing_logs/` are training logs and are not required for inference.
- `unicell_v1.ep*.pth` stores only a raw state dict and does not contain complete metadata, so it cannot directly replace `unicell_v1.best.pth`.
- For an `input_type="expr"` model, the seven-file set above is the complete inference artifact.
- `.pth` and `.pk` use Python/PyTorch deserialization mechanisms, which may execute malicious code. Load checkpoints only from trusted sources, and verify their SHA256 before loading.

### Portable GeneFormer/scGPT Checkpoints

A foundation-backed checkpoint created by `unicell-train-backbone` additionally
contains `run_config.json` and a lightweight reconstruction bundle:

```text
checkpoint_directory/
├── seven base files listed above
├── run_config.json
└── backbone/
    ├── manifest.json
    ├── config.json + gene_vocab.json    # GeneFormer
    └── args.json + vocab.json           # scGPT
```

Only the pair for the selected backbone is present. `unicell_v1.best.pth`
contains the trained backbone parameters; `backbone/` contains only the small
configuration and vocabulary needed to reconstruct its architecture. It does
not copy the original GeneFormer/scGPT pretrained weights. Move the complete
checkpoint directory and pass its new location as `--ckpt_dir`. Legacy
checkpoints may still contain absolute `llm_model_file`, `llm_vocab_file`, and
`llm_args_file` paths; override them with the inference options below.

## H5AD Input Requirements

### Inference Data

- The input must be a `.h5ad` file readable by `anndata.read_h5ad()`.
- `adata.var_names` should use the same gene identifier convention as `gene_names.pk`, usually gene symbols. At least one gene must match the checkpoint, or inference will fail.
- Inference reconstructs the expression matrix in checkpoint gene order: extra input genes are ignored, and missing genes are filled with zeros.
- `unicell_predict()` applies the same heuristic preprocessing used by the training/evaluation reader to a copy of the input. For a nonnegative matrix with more than 1,000 genes it filters cells with fewer than 200 detected genes. If the maximum value is greater than 25, it applies `log1p`; integer-like input is first normalized to a total count of 1e4. Data already normalized and log-transformed is not normalized or log-transformed again when its maximum is at most 25, although the cell filter still applies.
- If only predictions are needed, `obs` does not need to contain ground-truth cell type/tissue/species columns; the corresponding metrics are simply skipped when these columns are absent.
- To compute metrics, the default ground-truth label columns are:
  - `cell_type_ontology_term_id`
  - `general_tissue`
  - `organism`
- Chunked inference additionally requires `obs` to contain the column specified by `--sample_key`, which defaults to `sample`.

> [!CAUTION]
> Inference creates a new AnnData with genes reordered according to the checkpoint. The original `var`, `layers`, `raw`, and other contents are not preserved in full. If these contents will still be needed later, retain the original H5AD first.

### Training and Validation Data

The default training entry point requires both the training and validation datasets to contain the following `obs` columns:

```text
cell_type_ontology_term_id
general_tissue
organism
```

In addition:

- Cell types should use Cell Ontology IDs recognized by the model ontology.
- The validation set must contain every gene required by the training set; the code reorders the validation set into training gene order.
- The validation set must not contain tissue or species labels that never occur in the training set.
- HVG-selection code in the current training input pipeline is disabled; do not assume that the program automatically performs HVG selection.
- `scDataset.read_data()` in the training entry point still includes heuristic processing based on expression-value ranges: for nonnegative expression matrices with more than 1,000 genes, it filters out cells in which fewer than 200 genes are detected, and it may run `normalize_total(1e4)` and `log1p` on data that appears to contain raw counts. When publishing a checkpoint, document the preprocessing that was actually triggered.
- Before training, it is recommended to ensure that `obs_names` and `var_names` are unique and to explicitly document normalization, log1p, and gene-filtering procedures.

## Safe List: Optional Candidate-Label Constraints

A safe list affects only **final label decoding at inference time**. It does not participate in training and does not modify the raw logits saved in `obsm`. Without a safe list, the species, tissue, and cell-type classification heads each perform `argmax` over all classes independently. With a safe list, each cell is decoded in the following order:

```text
species -> tissue -> cell_type
```

Both `predict.py` and `predict_by_sample_chunks.py` call the same `unicell_predict()` function, so their safe-list rules are identical.

### Can It Be Omitted or Left Empty?

The safe list is not a required file. The recommended way to run without constraints is:

- Omit `--safe_list` entirely from the CLI.
- Set `safe_list_path=None` in the Python API.

A valid JSON object `{}` or the following content is also currently equivalent to no constraints:

```json
{
  "global": {},
  "hierarchy": {}
}
```

Do not pass a zero-byte empty file, because an empty file is not valid JSON and triggers `JSONDecodeError`. Also, do not use an empty array `[]` to mean “deny all.” The current implementation does not support deny-all behavior; an empty candidate set falls back to unconstrained `argmax`.

### Complete JSON Format

The repository's [`safe_list.json`](safe_list.json) is a format example:

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

| Field | Type | Meaning |
|---|---|---|
| `global.species` | Array of strings | Globally allowed species candidates |
| `global.tissue` | Array of strings | Globally allowed tissue candidates |
| `global.cell_type` | Array of strings | Globally allowed cell-type candidates |
| `hierarchy.<species>.tissue` | JSON object | Tissues allowed when the predicted species is this species; the object's keys are the tissue names |
| `hierarchy.<species>.tissue.<tissue>.cell_type` | Array of strings | Cell types allowed when this “species + tissue” combination has been predicted |

The top level must be a JSON object; both `global` and `hierarchy` may be omitted. Every name must **exactly match** a key in the current checkpoint dictionaries, including capitalization and spaces:

- Look up species in `species_dict.pk`.
- Look up tissues in `tissue_dict.pk`.
- Look up cell types in `celltype_dict.pk`. These should normally be `CL:...` IDs rather than natural-language display names.
- The safe-list field name is fixed as `species`; this is a different concept from the H5AD default species-column name, `organism`.

### Constraining Only One Dimension

Constrain species only:

```json
{
  "global": {
    "species": ["Homo sapiens"]
  }
}
```

Constrain tissue only:

```json
{
  "global": {
    "tissue": ["brain", "lung"]
  }
}
```

Constrain cell type only:

```json
{
  "global": {
    "cell_type": ["CL:0000540", "CL:0000127"]
  }
}
```

Dimensions not present in the configuration continue to select from all classes in the checkpoint. For example, specifying only `global.tissue` does not constrain species or cell type, nor does it automatically infer their biological relationships with tissues.

### Hierarchical Constraints

The complete example above means:

1. First select a species from `global.species`.
2. If `Homo sapiens` is predicted, the tissue can only be selected from `brain` and `lung`.
3. If `Mus musculus` is predicted, the only selectable tissue is `brain`.
4. Finally, select a cell type from the corresponding “species + tissue” leaf node's `cell_type` list.

The presence of a species in `hierarchy` **does not constrain the species classification head**; the species level reads only `global.species`. Therefore, to apply strict hierarchical constraints:

- List every allowed species in a nonempty `global.species`.
- Configure at least one checkpoint-recognized tissue for every allowed species.
- Configure at least one checkpoint-recognized cell type for every species–tissue combination that needs to be constrained.

### How `global` and `hierarchy` Are Combined

The species level uses only:

```text
global.species
```

When constraints exist on both sides, the tissue level uses:

```text
global.tissue ∩ hierarchy[predicted species].tissue.keys()
```

When constraints exist on both sides, the cell-type level uses:

```text
global.cell_type
∩ hierarchy[predicted species].tissue[predicted tissue].cell_type
```

Combination rules:

- If both sides exist, their intersection is used.
- If only one side exists, that side is used.
- If neither side exists, that level is unconstrained.
- If the predicted species has no hierarchy branch, the tissue level applies any remaining global constraint only.
- If a “species + tissue” combination has no `cell_type` configuration, the cell-type level applies any remaining global constraint only.

For example, although `global.tissue` in the repository example includes `heart`, neither of the two species branches in `hierarchy` contains `heart`, so their intersection does not allow `heart`.

### Current Behavior for Empty Values, Unknown Labels, and Conflicts

| Configuration | Current implementation behavior |
|---|---|
| Field omitted | That source applies no constraint |
| `{}`, empty `global`, or empty `hierarchy` | The corresponding part applies no constraint |
| An array contains both known and unknown labels | Prints `[SafeList][WARN]`; known labels remain effective |
| An array is `[]` | The empty candidate set ultimately falls back to an all-class `argmax` for that classification head |
| Every label in an array is unknown | Produces an empty candidate set after the warning, then falls back to an all-class `argmax` |
| The intersection of global and hierarchy is empty | Falls back to an all-class `argmax`; the result may even fall outside both lists |
| A species in hierarchy has no valid tissue | No tissue constraint is created for that species |
| A tissue omits `cell_type` or sets it to `null` | That species–tissue combination applies no hierarchical cell-type constraint |
| Incorrect JSON types | May directly trigger `TypeError` or `AttributeError` |

Therefore, before publishing a safe list, check that:

- Empty arrays are not used to express “deny all.”
- Every array contains at least one label known to the corresponding checkpoint.
- Every global/hierarchy intersection that must apply simultaneously is nonempty.
- Arrays are written as JSON arrays; do not write a single value as an ordinary string.

The following script displays the exact labels supported by a checkpoint. Pickle files must likewise be loaded only from trusted sources:

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

## Usage After Installing the Python Package

Installing the wheel or source package provides four commands:

| Command | Purpose |
|---|---|
| `unicell-predict` | Run inference on a single H5AD file |
| `unicell-predict-chunks` | Run per-sample chunked inference on a large H5AD file |
| `unicell-train` | Compatibility entry point for the original expression-matrix training workflow |
| `unicell-train-backbone` | Validated configurable training with an expr, GeneFormer, or scGPT encoder |

### Single-File Inference CLI

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

If no constraints are needed, simply remove `--safe_list`. The parameters are as follows:

| Parameter | Required/Default | Description |
|---|---|---|
| `--input` | Required | Input `.h5ad` file |
| `--ckpt_dir` | Required | Checkpoint directory containing the seven required files |
| `--output` | `<input>_annotated.h5ad` | Output H5AD path |
| `--device` | Automatically selected | Either `cuda` or `cpu`; if omitted, CUDA is used when available, otherwise CPU is used |
| `--batch_size` | `2048` | Inference batch size; reduce it if GPU memory is insufficient |
| `--safe_list` | Not used | Optional path to a safe-list JSON file |
| `--backbone_model` | Checkpoint metadata | Override a legacy GeneFormer model directory or scGPT `.pt` path |
| `--backbone_vocab` | Checkpoint metadata | Override a legacy GeneFormer/scGPT vocabulary path |
| `--backbone_args` | Checkpoint metadata | Override a legacy scGPT `args.json` path |
| `--init_backbone_from_config` | Checkpoint metadata | Reconstruct the backbone from configuration before loading the complete UniCell state dict; intended for migration |
| `--save_metrics_json` | Disabled | Save metrics to `<output_stem>_metrics.json` |
| `--cell_type_key` | `cell_type_ontology_term_id` | Ground-truth cell-type column in `obs`, used to compute metrics |
| `--tissue_key` | `general_tissue` | Ground-truth tissue column in `obs`, used to compute metrics |
| `--species_key` | `organism` | Ground-truth species column in `obs`, used to compute metrics |

Whenever the corresponding ground-truth label columns are present, the command prints the relevant metrics to the terminal; `--save_metrics_json` only controls whether an additional JSON file is saved.

### Per-Sample Chunked Inference CLI for Large Files

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

| Parameter | Required/Default | Description |
|---|---|---|
| `--input` | Required | Input large `.h5ad` file |
| `--ckpt_dir` | Required | Checkpoint directory |
| `--out_dir` | Required | Directory for each sample's H5AD/CSV outputs |
| `--sample_key` | `sample` | `obs` column used to split the data |
| `--device` | `cuda` | `cuda` or `cpu`; automatically falls back to CPU if CUDA is requested but unavailable |
| `--batch_size` | `2048` | Inference batch size within each chunk |
| `--safe_list` | Not used | Optional path to a safe-list JSON file |
| `--backbone_model` | Checkpoint metadata | Override a legacy GeneFormer model directory or scGPT `.pt` path |
| `--backbone_vocab` | Checkpoint metadata | Override a legacy GeneFormer/scGPT vocabulary path |
| `--backbone_args` | Checkpoint metadata | Override a legacy scGPT `args.json` path |
| `--init_backbone_from_config` | Checkpoint metadata | Reconstruct the backbone from configuration before loading the complete UniCell state dict; intended for migration |
| `--compression` | `gzip` | H5AD compression method: `gzip` or `none` |
| `--compression_opts` | `4` | gzip compression level |
| `--skip_done` | Disabled | Skip a chunk when both its H5AD and CSV files already exist |
| `--save_metrics_json` | Disabled | Save a metrics JSON file for each chunk |
| `--summary_csv` | `<out_dir>/chunk_summary.csv` | Custom path for the summary CSV |
| `--cell_type_key` | `cell_type_ontology_term_id` | Ground-truth cell-type column in `obs` |
| `--tissue_key` | `general_tissue` | Ground-truth tissue column in `obs` |
| `--species_key` | `organism` | Ground-truth species column in `obs` |

Each sample produces an annotated H5AD file and a prediction CSV. A summary CSV is written separately after the job finishes.

### Legacy Expression-Only Training CLI

`unicell-train` preserves the original fixed `input_type="expr"` workflow.
Use `unicell-train-backbone` below for configurable expression, GeneFormer, or
scGPT training; the new command does not change the legacy entry point.

Single GPU, or automatic CPU/GPU selection:

```bash
unicell-train \
  --train_h5ad ./data/train.h5ad \
  --eval_h5ad ./data/valid.h5ad
```

Multi-GPU DDP:

```bash
torchrun --standalone --nproc_per_node=2 --module train_Unicell \
  --train_h5ad ./data/train.h5ad \
  --eval_h5ad ./data/valid.h5ad
```

Currently, `unicell-train` has only two CLI parameters:

| Parameter | Required | Description |
|---|---|---|
| `--train_h5ad` | Yes | Training-set H5AD file |
| `--eval_h5ad` | Yes | Validation-set H5AD file |

The remaining settings are not CLI parameters. In `train_Unicell.py`, the uppercase items below are module-level constants, while `input_type` and `output_dim` are hard-coded arguments in the Trainer construction:

| Configuration | Current Value | Purpose |
|---|---|---|
| `PREFIX` | `checkpoints` | Experiment/output-directory suffix |
| `CKPT_DIR` | `models/checkpoints` | Checkpoint output directory, relative to the current working directory |
| `BATCH_SIZE` | `128` | Training batch size |
| `LEARNING_RATE` | `1e-3` | Adam learning rate |
| `NUM_EPOCHS` | `50` | Number of epochs |
| `BETA` | `0.1` | Weighting parameter for the combined loss in the Trainer |
| `GLOBAL_LAYER` | `128` | HMCN global hidden size |
| `LOCAL_LAYER` | `64` | HMCN local hidden size |
| `HIDDEN_LAYER_DROPOUT` | `0.1` | Hidden-layer dropout |
| `CELL_TYPE_KEY` | `cell_type_ontology_term_id` | Cell-type label column |
| `TISSUE_KEY` | `general_tissue` | Tissue label column |
| `SPECIES_KEY` | `organism` | Species label column |
| `input_type` | `expr` | The current entry point uses a standard expression matrix |
| `output_dim` | `512` | Encoder output dimension |

To customize the uppercase settings, modify the source constants and reinstall, or override the module configuration in Python before calling it as shown below. Changing `input_type` or `output_dim` requires editing the Trainer construction or using the low-level `UnicellTrainer` API.

### Foundation-Encoder Training CLI (GeneFormer/scGPT)

`unicell-train-backbone` is the configurable training entry point introduced in
the 0.2.0 wheel. Training and validation H5AD files must contain
`cell_type_ontology_term_id`, `general_tissue`, and `organism`. The command
always performs startup validation; append `--validate_only` to stop after
checking paths, schemas, gene overlap, vocabulary compatibility, checkpoint
structure, device selection, and cells per DDP rank.

GeneFormer uses the vocabulary bundled in UniCell by default. Point
`--llm_model_file` to a local Hugging Face model directory containing
`config.json` and model weights:

```bash
unicell-train-backbone \
  --input_type GeneFormer \
  --train_h5ad ./data/train.h5ad \
  --eval_h5ad ./data/valid.h5ad \
  --llm_model_file /models/Geneformer-V1-10M \
  --ckpt_dir ./models/unicell_geneformer \
  --validate_only
```

For scGPT, the checkpoint, vocabulary, and `args.json` must come from the same
pretrained release:

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

Remove `--validate_only` to train. For DDP, launch the packaged module with
`torchrun --standalone --nproc_per_node=N -m unicell.cli.train_backbone` and
the same arguments. Each process must receive at least one full batch because
the existing Trainer uses `drop_last=True`.

The wheel does not redistribute foundation-model weights. During training they
are loaded from the paths above. A successful run writes the seven base files,
`run_config.json`, and the portable `backbone/` bundle documented under
[Checkpoint Files](#checkpoint-files). Move the entire checkpoint directory,
then run inference without the original pretrained-model path:

```bash
unicell-predict \
  --input ./data/test.h5ad \
  --ckpt_dir ./models/moved_unicell_scgpt \
  --output ./results/test_annotated.h5ad \
  --device cuda
```

For a legacy GeneFormer/scGPT checkpoint whose metadata points to another
machine, pass `--backbone_model`, `--backbone_vocab`, and, for scGPT,
`--backbone_args`:

```bash
# Legacy GeneFormer checkpoint
unicell-predict --input ./data/test.h5ad \
  --ckpt_dir ./models/legacy_geneformer \
  --backbone_model /models/Geneformer-V1-10M \
  --output ./results/geneformer_annotated.h5ad

# Legacy scGPT checkpoint
unicell-predict --input ./data/test.h5ad \
  --ckpt_dir ./models/legacy_scgpt \
  --backbone_model /models/scgpt/best_model.pt \
  --backbone_vocab /models/scgpt/vocab.json \
  --backbone_args /models/scgpt/args.json \
  --output ./results/scgpt_annotated.h5ad
```

`--init_backbone_from_config` is an advanced migration option for reconstructing
an architecture before loading a complete compatible UniCell state dict. The
same four options are supported by `unicell-predict-chunks`. Standard
`input_type="expr"` checkpoints do not enter a foundation-encoder branch.

### Python Inference API

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

The return value is an `scDataset`, not an `AnnData` object directly; the annotated object is available as `result.adata`.

Parameters for `unicell_predict()`:

| Parameter | Default | Description |
|---|---|---|
| `adata` | None | Required AnnData object |
| `filepath` | None | Required input file identifier/path; this is usually the path of the H5AD file from which the AnnData object was read |
| `ckpt_dir` | Syntactic default: `None`; required in practice | Checkpoint directory; omitting it causes an error when model paths are constructed |
| `batch_size` | `512` | Python API inference batch size |
| `device` | `None` | When `None`, CUDA/CPU is selected automatically; `"cuda"` or `"cpu"` can also be passed explicitly |
| `cell_type_key` | `cell_type_ontology_term_id` | Ground-truth cell-type column used for metrics |
| `tissue_key` | `general_tissue` | Ground-truth tissue column used for metrics |
| `species_key` | `organism` | Ground-truth species column used for metrics |
| `compute_metrics` | `True` | Compute metrics for ground-truth label columns that are present; tasks with missing columns are skipped |
| `safe_list_path` | `None` | Path to a safe-list JSON file; `None` means unconstrained inference |
| `backbone_model_file` | `None` | Override the foundation-model path stored by a legacy checkpoint |
| `backbone_vocab_file` | `None` | Override the foundation vocabulary stored by a legacy checkpoint |
| `backbone_args_file` | `None` | Override a legacy scGPT `args.json` path |
| `initialize_backbone_from_config` | Checkpoint metadata | Reconstruct the foundation architecture from its config before loading the complete UniCell state dict |

### Python Training API

The wheel installs the top-level `train_Unicell` module, allowing direct use of the packaged data-validation and training workflow:

```python
import torch
import train_Unicell as train_entry

# Optional: override module-level configuration before calling.
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

The direct parameters of `train_unicell()` are limited to:

| Parameter | Description |
|---|---|
| `ddp_train` | Whether to train in DDP mode; use `False` for a standard Python call |
| `local_rank` | Local process/GPU rank; use `0` for a standard Python call |
| `device` | For example, `cuda`, `cuda:0`, or `cpu` |
| `train_h5ad` | Training-set path |
| `eval_h5ad` | Validation-set path |

Module constants are the current configuration mechanism for this entry point, not a stable high-level hyperparameter API. For full control over the model, use `unicell.trainer.UnicellTrainer` directly. Its constructor parameters include:

```text
scDataset, input_type, input_dim, output_dim,
batch_size, learning_rate, num_epochs, beta, device,
global_layer, local_layer, hidden_layer_dropout, ckpt_dir,
ddp_train=False, save_epoch=False, local_rank=0,
llm_model_file=None, llm_vocab_file=None, llm_args_file=None
```

The low-level API requires callers to construct and align `scDataset` correctly themselves, and to save `gene_names.pk`, `ontoGraph.pk`, and `ontoGraph.graph.gml`. `UnicellTrainer` adds only the model weights and the three label dictionaries; if any ontology or gene files are omitted, the generated directory cannot be used directly for inference. Prefer `train_Unicell.train_unicell()` unless you genuinely need to customize the low-level training workflow.

## Running the Repository's Python Files Directly

After installing the dependencies from the repository root, the core scripts correspond to the installed commands:

```bash
# Parameters are exactly the same as those for unicell-predict
python predict.py \
  --input ./data/input.h5ad \
  --ckpt_dir ./models/unicell_expression_v1 \
  --output ./results/input_annotated.h5ad \
  --device cuda \
  --batch_size 2048 \
  --safe_list ./safe_list.json \
  --save_metrics_json

# Parameters are exactly the same as those for unicell-predict-chunks
python predict_by_sample_chunks.py \
  --input ./data/atlas.h5ad \
  --ckpt_dir ./models/unicell_expression_v1 \
  --out_dir ./results/atlas_chunks \
  --sample_key sample \
  --skip_done

# Parameters are exactly the same as those for unicell-train
python train_Unicell.py \
  --train_h5ad ./data/train.h5ad \
  --eval_h5ad ./data/valid.h5ad

# Configurable expr/GeneFormer/scGPT training from source
python -m unicell.cli.train_backbone --help
```

Multi-GPU execution from source:

```bash
torchrun --standalone --nproc_per_node=2 train_Unicell.py \
  --train_h5ad ./data/train.h5ad \
  --eval_h5ad ./data/valid.h5ad
```

The complete parameters match the corresponding tables in the previous section. You can also view them directly:

```bash
python predict.py --help
python predict_by_sample_chunks.py --help
python train_Unicell.py --help
python -m unicell.cli.train_backbone --help
```

### Changing the Training Set Every Epoch

The directory must be organized as follows:

```text
seed_root/
├── seed_0/train.h5ad
├── seed_1/train.h5ad
├── ...
└── seed_49/train.h5ad
```

Run:

```bash
python train_epoch_unicell.py \
  --eval_h5ad ./data/valid.h5ad \
  --seed_root ./data/sampled_h5ads \
  --seed_start 0
```

| Parameter | Required/Default | Description |
|---|---|---|
| `--eval_h5ad` | Required | Validation set shared by all epochs |
| `--seed_root` | Required | Root directory containing `seed_i/train.h5ad` |
| `--seed_start` | `0` | Seed number for the first epoch; epoch `e` uses `seed_start + e` |

This script always runs for 50 epochs, reading files in order from `<seed_root>/seed_<seed_start>/train.h5ad` through `<seed_root>/seed_<seed_start+49>/train.h5ad`. The output directory is fixed as `models/last_version_gml`. It is not a wheel console command; changing the number of epochs or output directory requires editing the constants at the top of the script.

### Inference and UMAP Plotting

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

| Parameter | Required/Default | Description |
|---|---|---|
| `--test_h5ad` | Required | Input H5AD file |
| `--ckpt_dir` | Required | Checkpoint directory |
| `--out_dir` | `./unicell_umap` | Output directory for figures, metadata, and the optional H5AD file |
| `--species_col` | `organism` | Species column used for coloring/metrics |
| `--tissue_col` | `general_tissue` | Tissue column used for coloring/metrics |
| `--celltype_col` | `cell_type_ontology_term_id` | Cell-type column used for coloring/metrics |
| `--shared_emb_key` | `unicell_emb` | `obsm` key for the shared embedding |
| `--species_head_key` | `species_cls_emb` | `obsm` key for species logits |
| `--tissue_head_key` | `tissue_cls_emb` | `obsm` key for tissue logits |
| `--celltype_head_key` | `cls_emb` | `obsm` key for cell-type logits |
| `--device` | Automatically selected | `cuda` or `cpu` |
| `--batch_size` | `2048` | Inference batch size |
| `--no_metrics` | Disabled | When passed, disables metrics inside `unicell_predict()` |
| `--n-neighbors` | `30` | Number of UMAP neighbors |
| `--min-dist` | `0.5` | UMAP minimum distance |
| `--random-state` | `0` | Random seed |
| `--legend` | Disabled | Display legends in the plots |
| `--pt-size` | `4.0` | Scatter-point size |
| `--alpha` | `0.8` | Scatter-point opacity |
| `--no_write_h5ad` | Disabled | When passed, do not save the H5AD file containing the embedding/UMAP |

`run_umap.py` currently has no `--safe_list` parameter, so it performs unconstrained inference internally.

## Inference Outputs

Standard inference writes the following into the resulting AnnData object:

| Location | Key | Contents |
|---|---|---|
| `obs` | `predicted_cell_type_ontology_id` | Predicted Cell Ontology ID |
| `obs` | `predicted_cell_type` | String representation currently identical to the predicted ontology ID |
| `obs` | `predicted_tissue` | Predicted tissue |
| `obs` | `predicted_species` | Predicted species |
| `obs` | `level_0`, `level_1`, ... | Ontology path for the predicted cell type, from the common ancestor to the leaf node |
| `obsm` | `unicell_emb` | Shared Encoder representation |
| `obsm` | `cls_emb` | Cell-type classification-head logits |
| `obsm` | `tissue_cls_emb` | Tissue classification-head logits |
| `obsm` | `species_cls_emb` | Species classification-head logits |
| `uns` | `cls_cell_type` | Cell-type order corresponding to the columns of `cls_emb` |
| `uns` | `cls_tissue` | Tissue order corresponding to the columns of `tissue_cls_emb` |
| `uns` | `cls_species` | Species order corresponding to the columns of `species_cls_emb` |

If the corresponding ground-truth labels are present in the input `obs`, standard inference computes Accuracy and Macro-F1. Chunked inference can also save a JSON file for each chunk and aggregate the results into a CSV.

## Documentation, Notebooks, and Release Notes

Further resources:

- [`docs/API/`](docs/API/): Command-line, `scDataset`, `OntoGRAPH`, `UnicellTrainer`, `unicell_predict`, and metrics APIs.
- [`docs/PACKAGE ORGANIZATION/`](docs/PACKAGE%20ORGANIZATION/): Model architecture and HMCN loss.
- [`docs/TUTORIALS/`](docs/TUTORIALS/): Tutorials on ontology-guided annotation, novel cell-type detection, naming harmonization, atlas construction, and supervised annotation with GeneFormer/scFoundation/scGPT.
- [`predict.ipynb`](predict.ipynb): Standard inference notebook.
- [`predict_by_sample_chunks.ipynb`](predict_by_sample_chunks.ipynb): Chunked-inference notebook for large files.
- [`unicell-train-predicct.ipynb`](unicell-train-predicct.ipynb): Training and inference notebook.

### Data and Artifact Policy

Do not commit patient-level data, H5AD datasets, checkpoints, prediction outputs, logs, wheels, or offline dependency packages to Git. Approved models should be distributed through a model/Release service with versions and checksums; datasets should be distributed through data repositories with appropriate access controls and licenses.

### Third-Party Code and License Status

`unicell/repo/` contains adapted or embedded code related to multiple single-cell foundation-model projects. Read [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistributing it.

Original UniCell contributions are available under the [MIT License](LICENSE), Copyright (c) 2026 UniCell contributors. Incorporated third-party code and data retain their own licenses, including GPL v3, Apache-2.0, MIT/BSD, and CC BY 4.0; the complete distribution is not MIT-only. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/) for attribution, source references, and the remaining SCAD/DSBN documentation gaps. Code licenses do not automatically cover separately downloaded model weights or datasets.
