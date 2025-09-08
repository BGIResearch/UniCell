# `unicell_predict` Function

This module provides the inference interface for predicting hierarchical cell types using a trained UniCell model. It loads the saved model checkpoint, processes input `AnnData`, and annotates cells with predicted ontology-aligned types.

---

## 🧬 Function: `unicell_predict`

```python
unicell_predict(
    adata,
    filepath,
    ckpt_dir=None,
    batch_size=512,
    device=None
)
```

### Description
Loads a trained UniCell model and predicts cell type labels and embeddings for new input single-cell data.

### Parameters
- **adata** : `AnnData`  
  Single-cell expression data to annotate. Can be raw or normalized.

- **filepath** : `str`  
  Path to original `.h5ad` file, used to assist with reprocessing and gene matching.

- **ckpt_dir** : `str`, optional  
  Directory containing model checkpoint (`unicell_v1.best.pth`), vocabulary, and metadata.

- **batch_size** : `int`, default: `512`  
  Mini-batch size for inference.

- **device** : `str`, optional  
  Target computation device, such as `"cuda"` or `"cpu"`.

### Returns
- **scDataset** : `scDataset`  
  Updated dataset containing:
  - `.obs['predicted_cell_type']`
  - `.obs['predicted_cell_type_ontology_id']`
  - `.obsm['unicell_emb']` for cell embeddings
  - `.obsm['cls_emb']` for classifier layer logits
  - `.uns['cls_cell_type']` with cell type labels
  - Hierarchical annotations across multiple levels

---

## 🔍 Helper Function: `read_data`

```python
read_data(adata, filepath, ckpt_dir, llm_vocab, llm_args)
```

### Description
Prepares the input data by aligning genes to the trained model vocabulary and re-encoding expression matrix. Handles sparse input and ontology graph loading.

### Returns
- `scDataset`: Normalized, filtered, and ontology-linked dataset.

---

## 🔍 Helper Function: `collate_fn_with_args`

```python
collate_fn_with_args(input_type, model)
```

### Description
Custom collation logic to format batch data based on the input type.

- For `"GeneFormer"` input, tokenizes and pads sequences.
- For others, returns tensor batches directly.

Returns:
- `callable`: DataLoader-compatible collate function

---

## ✅ Example

```python
from unicell.anno_predict import unicell_predict
import anndata as ad

adata = ad.read_h5ad("example_input.h5ad")
annotated_dataset = unicell_predict(
    adata=adata,
    filepath="example_input.h5ad",
    ckpt_dir="./checkpoints",
    device="cuda"
)

annotated_dataset.adata.obs["predicted_cell_type"].head()
```