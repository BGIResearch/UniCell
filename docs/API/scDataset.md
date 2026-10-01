# `scDataset` Module

This module preprocesses and manages single-cell gene expression datasets for ontology-based training or inference. In the current multi-task model, it also prepares tissue and organism labels alongside the Cell Ontology hierarchy.

---

## 🧬 Class: `scDataset`

```python
scDataset(
    adata=None,
    data_path=None,
    cell_type_key=None,
    tissue_key=None,
    species_key=None,
    batch_key=None,
    trained=False,
    graph=None,
    highly_variable_genes=True,
    llm_vocab=None,
    llm_args=None
)
```

### Description
Initializes a single-cell dataset from a given `.h5ad` file or AnnData object. If `trained=True`, it builds a Cell Ontology subgraph, filters unsupported cell-type labels, and creates class indices for cell type plus tissue and organism when their corresponding keys are supplied.

### Parameters
- **adata** : `AnnData`, optional  
  Pre-loaded AnnData object. If not provided, `data_path` is used.

- **data_path** : `str`, optional  
  Path to `.h5ad` file. Required if `adata` is not provided.

- **cell_type_key** : `str`, optional  
  Key in `.obs` containing Cell Ontology IDs. Required if `trained=True` and when evaluating known cell types.

- **tissue_key** : `str`, optional  
  Key in `.obs` containing tissue labels. The current trainer requires this key; the project training entry point uses `"general_tissue"`.

- **species_key** : `str`, optional  
  Key in `.obs` containing organism labels. The current trainer requires this key; the project training entry point uses `"organism"`. The name `species_key` is retained by the implementation.

- **batch_key** : `str`, optional  
  Key in `.obs` representing batch metadata. Used to create batch index.

- **trained** : `bool`, default: `False`  
  Whether to apply ontology-based training preprocessing.

- **graph** : `networkx.Graph`, optional  
  Pre-loaded ontology graph. If omitted during training, the packaged `graph.gml` resource is loaded.

- **highly_variable_genes** : `bool`, default: `True`  
  Compatibility argument passed to `read_data`. Highly variable gene subsetting is disabled in the current code.

- **llm_vocab** : `str`, optional  
  Path to a JSON or pickle gene vocabulary used by a foundation-model input pipeline.

- **llm_args** : `str`, optional  
  JSON path specifying additional LLM arguments.

---

### 🔧 Internal Attributes
- `adata`: Preprocessed `AnnData` object.
- `ontograph`: OntoGRAPH instance if `trained=True`.
- `batch_index`: Batch indices if `batch_key` is provided.
- `cell_type_index`: List of indices into the ontology vocabulary.
- `tissue_label_dict`, `tissue_index`: Tissue-to-index mapping and per-cell tissue indices if `tissue_key` is provided.
- `species_label_dict`, `species_index`: Organism-to-index mapping and per-cell organism indices if `species_key` is provided.
- `llm_vocab`, `llm_args`: LLM configuration and vocabulary for gene models.

---

## 🔍 Methods

### `read_data(highly_variable_genes=True, subset=True)`
Loads and preprocesses an `.h5ad` dataset.

- This method is called only when `adata` is not supplied.
- For non-negative matrices, it filters cells with fewer than 200 detected genes when more than 1,000 genes are present.
- If the maximum value is greater than 25, it applies `log1p`; integer-like input is first normalized to a total count of 1e4.
- It does not currently subset highly variable genes.

Returns: `AnnData`

---

### `get_cell_type_index()`
Maps each cell's type to an index in the ontology vocabulary.

Returns: `list[int]`

---

### `get_initial_embeddings(input_dim, adata_trained=None, emb_key=None)`
Generates PCA-based embeddings.

- If `trained=True`, fits PCA.
- Otherwise, projects onto PCA space of `adata_trained`.

Returns: `np.ndarray`

---

### `get_cell_type_hierarchy_matrix()`
Builds label matrices for hierarchical classification.

Returns: `list[torch.Tensor]`

---

## ✅ Example

```python
from unicell.scDataset import scDataset

dataset = scDataset(
    data_path="data.h5ad",
    cell_type_key="cell_type_ontology_term_id",
    tissue_key="general_tissue",
    species_key="organism",
    trained=True
)

emb = dataset.get_initial_embeddings(input_dim=50)
hier_labels = dataset.get_cell_type_hierarchy_matrix()
```

The three label columns are required by the current `UnicellTrainer` and must be non-missing for every training cell. Validation data must reuse the training dataset's `tissue_label_dict` and `species_label_dict` so that auxiliary-task class indices remain aligned; validation tissue or organism values not seen during training must be rejected.
