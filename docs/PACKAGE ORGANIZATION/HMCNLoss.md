
# ⚖️ HMCNLoss Functions for UniCell

This document describes the loss functions used in the current UniCell multi-task framework for Cell Ontology hierarchy learning and flat cell-type, tissue, and organism classification.

---

## 🧩 HMCNLoss

The `HMCNLoss` (Hierarchical Multi-Label Classification Network Loss) supports **scGPT** and all other input types. It calculates:

- **Global Loss**: Classification at different hierarchical levels.
- **Local Loss**: Fine-grained classification within each hierarchical level.

### Parameters

- `input_type`: Input data type (`expr`, `scGPT`, etc.)
- `scDataset`: Dataset object, must contain ontology graph

### Forward Logic

```python
global_loss, local_loss = HMCNLoss(input_type, scDataset)(
    global_layer_output, local_layer_outputs, batch_labels
)
```

### Behavior by Input Type

- If `input_type == 'scGPT'`:
  - Uses `F.cross_entropy` for both global and local layers.
- Otherwise:
  - Uses `F.binary_cross_entropy` for multilabel targets at each level.

---

## 🔥 FocalLoss

The `FocalLoss` focuses training on hard examples. The current trainer instantiates separate focal losses for the flat cell-type, tissue, and organism heads.

### Parameters

- `num_classes`: Total number of classes
- `alpha`: Scaling factor (default: 1)
- `gamma`: Focusing parameter (default: 2)
- `reduction`: Either `"mean"` or `"sum"`

### Usage

```python
loss_fn = FocalLoss(num_classes=..., alpha=1, gamma=2)
loss = loss_fn(inputs, targets)
```

### Combined Training Objective

The current trainer combines the hierarchy and auxiliary tasks as:

```python
loss = (
    beta * global_loss
    + local_loss
    + global_cls_loss
    + tissue_cls_loss
    + species_cls_loss
)
```

Here `species_cls_loss` is the organism-classification loss. `beta` weights only the global hierarchy loss.

---

## 🧬 MMD Loss

The `compute_mmd` function calculates **Maximum Mean Discrepancy (MMD)** between different batches. It is available as a helper for batch effect correction and domain adaptation, but it is not included in the current `UnicellTrainer` objective.

### Signature

```python
mmd_loss = compute_mmd(features, batch_labels)
```

- `features`: Embeddings or outputs from the encoder.
- `batch_labels`: Tensor indicating batch membership.

---

## 📌 Summary

| Loss Function | Purpose | Applicable To |
|---------------|---------|---------------|
| `HMCNLoss` | Supervised hierarchical classification | All input types |
| `FocalLoss` | Optimize flat cell type, tissue, and organism heads | Current multi-task training |
| `compute_mmd` | Encourage inter-batch feature alignment | Optional; not used by the current trainer |

