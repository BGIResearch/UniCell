# `evaluate` Module

This module implements task-agnostic evaluation metrics for cell type, tissue, or organism classification, including accuracy and F1 scores. It aligns predictions and labels into a unified label space.

---

## 🧬 Function: `compute_metrics`

```python
compute_metrics(
    scDataset,
    label_key,
    prediction_key
)
```

### Description
Computes classification performance metrics including accuracy, macro F1, and micro F1 using `sklearn.metrics`. Handles alignment of predicted and true labels into a consistent categorical space.

### Parameters
- **scDataset** : `scDataset`  
  The annotated dataset containing both ground truth and predicted labels in `.obs`.

- **label_key** : `str`  
  Column name in `adata.obs` representing the true labels.

- **prediction_key** : `str`  
  Column name in `adata.obs` representing the predicted labels.

### Returns
- **metrics** : `dict`  
  A dictionary with:
  - `'accuracy'`: Standard classification accuracy.
  - `'macro_f1'`: F1 score averaged across all classes.
  - `'micro_f1'`: F1 score calculated globally.

---

## ✅ Example

```python
from unicell.evaluate import compute_metrics

metrics = compute_metrics(
    scDataset=scDataset,
    label_key="cell_type_ontology_term_id",
    prediction_key="predicted_cell_type_ontology_id"
)

print(metrics["macro_f1"])
```

The same function can evaluate the auxiliary tasks:

```python
tissue_metrics = compute_metrics(
    scDataset=scDataset,
    label_key="general_tissue",
    prediction_key="predicted_tissue"
)

organism_metrics = compute_metrics(
    scDataset=scDataset,
    label_key="organism",
    prediction_key="predicted_species"
)
```

The inference API uses a separate helper that skips missing ground-truth columns and unknown label pairs before reporting per-task accuracy and macro F1.
