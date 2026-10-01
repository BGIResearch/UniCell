# `UnicellTrainer` Module

This module handles training and evaluation of the HMCN (Hierarchical Multi-label Cell-type Network) model. It combines ontology-aware cell-type learning with flat cell-type, tissue, and organism classification heads, and supports distributed training and foundation-model input encodings.

---

## 🧬 Class: `UnicellTrainer`

```python
UnicellTrainer(
    scDataset,
    input_type,
    input_dim,
    output_dim,
    batch_size,
    learning_rate,
    num_epochs,
    beta,
    device,
    global_layer,
    local_layer,
    hidden_layer_dropout,
    ckpt_dir,
    ddp_train=False,
    save_epoch=False,
    local_rank=0,
    llm_model_file=None,
    llm_vocab_file=None,
    llm_args_file=None
)
```

### Description
Constructs a trainer object to handle multi-task training, loss computation, evaluation, and checkpointing. The supplied `scDataset` must contain non-missing Cell Ontology, tissue, and organism labels for every training cell.

### Parameters
- **scDataset** : `scDataset`  
  Preprocessed dataset instance containing expression, ontology mappings, `tissue_label_dict`, and `species_label_dict`. Construction fails if either auxiliary label dictionary is missing or empty.

- **input_type** : `str`  
  Input format. Supported values are `"expr"`, `"scFoundation"`, `"GeneFormer"`, and `"scGPT"`.

  Version 0.2.1 converts scFoundation rows to tensors in `HMCNDataset`, fixing the pandas `Series`/tensor mismatch during batching. Full scFoundation training and paper-result reproduction have not been verified for this release. The expression, GeneFormer, and scGPT input paths are unchanged.

- **input_dim** : `int`  
  Input feature dimensionality.

- **output_dim** : `int`  
  Output dimensionality.

- **batch_size** : `int`  
  Batch size used for training.

- **learning_rate** : `float`  
  Initial learning rate for optimizer.

- **num_epochs** : `int`  
  Total number of training epochs.

- **beta** : `float`  
  Weight applied to the global hierarchy loss. The current total loss is `beta * global_loss + local_loss + cell_type_loss + tissue_loss + species_loss`.

- **device** : `str`  
  Target compute device (e.g., `'cuda'` or `'cpu'`).

- **global_layer** : `int`  
  Hidden width used by each global hierarchy layer.

- **local_layer** : `int`  
  Hidden width used by each local hierarchy classifier.

- **hidden_layer_dropout** : `float`  
  Dropout applied to hidden layers.

- **ckpt_dir** : `str`  
  Directory to save model checkpoints.

- **ddp_train** : `bool`, default: `False`  
  Whether to use DistributedDataParallel for training.

- **save_epoch** : `bool`, default: `False`  
  If `True`, saves a raw model state dictionary as `unicell_v1.ep{epoch}.pth` after each epoch. These files do not contain the metadata required by the complete inference checkpoint.

- **local_rank** : `int`, default: `0`  
  Rank used for DDP training.

- **llm_model_file**, **llm_vocab_file**, **llm_args_file** : `str`, optional  
  Paths for LLM-specific model, vocab, and args for transformer input.

---

## 🔍 Method: `train(scdata_test=None)`

Trains the model for `num_epochs` and optionally evaluates on `scdata_test`. With a validation dataset, model selection uses flat cell-type macro F1; otherwise it uses training loss. The best model is saved to `unicell_v1.best.pth` together with metadata for the tissue and organism keys.

The trainer also writes:

- `celltype_dict.pk`
- `tissue_dict.pk`
- `species_dict.pk`
- `train_time_summary.json`
- per-epoch timing JSON files under `timing_logs/`

The project training entry point separately saves `gene_names.pk`, `ontoGraph.pk`, and `ontoGraph.graph.gml`, which are required for inference.

---

## 🔍 Method: `train_one_epoch(epoch)`

Executes a single training epoch.

- Computes global and local ontology losses plus focal losses for cell type, tissue, and organism.
- Supports DDP; CUDA automatic mixed precision is used for the scGPT forward path.

Returns: `float` (epoch loss)

---

## 🔍 Method: `predict(scDataset, batch_size)`

Runs validation inference through the flat cell-type head.

Returns:
- `dict`: Cell-type accuracy, macro F1, and micro F1. This method does not report tissue or organism metrics.

---

## 🔍 Method: `collate_fn(batch)`

Dynamic collation based on `input_type`.

- For GeneFormer, pads sequences and assembles token IDs.
- For `expr`, retrieves an expression batch from stored row indices.
- For other foundation-model inputs, returns the corresponding tensors or dictionaries.

Returns five fields when no batch key is present:

`(batch_data, batch_labels, cls_labels, tissue_labels, species_labels)`

When `batch_key` is present, `batch_batch_labels` is appended as a sixth field.

---

## 🧭 Project Training Workflows

The installed wheel exposes `unicell-train`, which accepts one training `.h5ad` and one validation `.h5ad`. Its current experiment configuration is fixed in `train_Unicell.py`: `input_type="expr"`, batch size 128, learning rate 1e-3, 50 epochs, `output_dim=512`, and output directory `models/checkpoints`.

The repository also contains `train_epoch_unicell.py`, the final experiment workflow used to rotate through seed-specific training files across epochs while keeping the first seed's genes, ontology, tissue classes, and organism classes fixed. Training and optimizer state continue across those files. This script is not included as a wheel console entry point and should not be confused with `unicell-train`.

Both workflows select the best checkpoint using flat cell-type macro F1 on validation data; tissue and organism predictions contribute to training but are not used for checkpoint selection.

---

## ✅ Example

```python
from unicell.trainer import UnicellTrainer
import os

ckpt_dir = "./checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

trainer = UnicellTrainer(
    scDataset=sc_train,
    input_type='expr',
    input_dim=sc_train.adata.shape[1],
    output_dim=128,
    batch_size=64,
    learning_rate=1e-4,
    num_epochs=10,
    beta=0.5,
    device='cuda',
    global_layer=2,
    local_layer=1,
    hidden_layer_dropout=0.2,
    ckpt_dir=ckpt_dir
)

trainer.train()
```

Here `sc_train` must have been constructed with, for example, `cell_type_key="cell_type_ontology_term_id"`, `tissue_key="general_tissue"`, and `species_key="organism"`.
