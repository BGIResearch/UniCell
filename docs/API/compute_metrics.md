# `evaluate` Module

This module implements evaluation metrics to assess single-cell classification performance, including accuracy and F1 scores. It aligns predictions and labels into a unified label space.

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
    label_key="cell_type",
    prediction_key="predicted_cell_type"
)

print(metrics["macro_f1"])
```