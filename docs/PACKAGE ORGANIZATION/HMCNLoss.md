
# ⚖️ HMCNLoss Functions for UniCell

This document describes the HMCNLoss functions used in the UniCell framework, which are crucial for optimizing hierarchical classification tasks in single-cell analysis.

---

## 🧩 HMCNLoss

The `HMCNLoss` (Hierarchical Multi-Label Classification Network Loss) supports both **scGPT** and non-LLM input types. It calculates:

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

The `FocalLoss` focuses training on hard examples, mitigating class imbalance, particularly useful for sparse labels in cell types.

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

---

## 🧬 MMD Loss

The `compute_mmd` function calculates **Maximum Mean Discrepancy (MMD)** between different batches. Useful for **batch effect correction** and domain adaptation.

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
| `FocalLoss` | Handle class imbalance | Flat classification problems |
| `compute_mmd` | Encourage inter-batch feature alignment | Multi-batch or domain-adaptation training |

