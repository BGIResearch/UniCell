# `UnicellTrainer` Module

This module handles training and evaluation of the HMCN (Hierarchical Multi-label Cell-type Network) model. It supports distributed training, ontology-aware loss functions, and LLM-based input encodings for single-cell data.

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
Constructs a trainer object to handle training, loss computation, optimizer scheduling, and checkpointing for hierarchical cell type prediction.

### Parameters
- **scDataset** : `scDataset`  
  Preprocessed dataset instance containing single-cell expression and ontology mappings.

- **input_type** : `str`  
  Input format. Options: `"pca"`, `"GeneFormer"`, `"scGPT"`.

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
  Weight factor for combining global and local hierarchical losses.

- **device** : `str`  
  Target compute device (e.g., `'cuda'` or `'cpu'`).

- **global_layer** : `int`  
  Number of global hierarchical layers.

- **local_layer** : `int`  
  Number of local layers used for refinement.

- **hidden_layer_dropout** : `float`  
  Dropout applied to hidden layers.

- **ckpt_dir** : `str`  
  Directory to save model checkpoints.

- **ddp_train** : `bool`, default: `False`  
  Whether to use DistributedDataParallel for training.

- **save_epoch** : `bool`, default: `False`  
  If `True`, saves model checkpoint after each epoch.

- **local_rank** : `int`, default: `0`  
  Rank used for DDP training.

- **llm_model_file**, **llm_vocab_file**, **llm_args_file** : `str`, optional  
  Paths for LLM-specific model, vocab, and args for transformer input.

---

## 🔍 Method: `train(scdata_test=None)`

Trains the model for `num_epochs` and optionally evaluates on `scdata_test`. Best model is saved to `unicell_v1.best.pth`.

---

## 🔍 Method: `train_one_epoch(epoch)`

Executes a single training epoch.

- Computes global, local, and class-level losses.
- Supports mixed-precision training and DDP.

Returns: `float` (epoch loss)

---

## 🔍 Method: `predict(scDataset, batch_size)`

Runs inference and evaluation.

Returns:
- `dict`: Evaluation metrics (e.g., accuracy, macro F1)

---

## 🔍 Method: `collate_fn(batch)`

Dynamic collation based on `input_type`.

- For GeneFormer, pads sequences and assembles token IDs.
- For others, returns stacked tensors or dictionaries.

Returns: `(batch_data, batch_labels, cls_labels)`

---

## ## ✅ Example

```python
from unicell.trainer import UnicellTrainer

trainer = UnicellTrainer(
    scDataset=scDataset,
    input_type='pca',
    input_dim=100,
    output_dim=128,
    batch_size=64,
    learning_rate=1e-4,
    num_epochs=10,
    beta=0.5,
    device='cuda',
    global_layer=2,
    local_layer=1,
    hidden_layer_dropout=0.2,
    ckpt_dir='./checkpoints'
)

trainer.train()
```