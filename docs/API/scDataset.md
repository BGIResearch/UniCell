# `scDataset` Module

This module preprocesses and manages single-cell gene expression datasets for ontology-based training or inference. It supports integration with ontology graphs, highly variable gene selection, and embedding generation.

---

## 🧬 Class: `scDataset`

```python
scDataset(
    adata=None,
    data_path=None,
    cell_type_key=None,
    batch_key=None,
    trained=False,
    graph=None,
    highly_variable_genes=True,
    llm_vocab=None,
    llm_args=None
)
```

### Description
Initializes and preprocesses a single-cell dataset from a given `.h5ad` file or AnnData object. If `trained=True`, applies ontology-based filtering and hierarchical annotation using `OntoGRAPH`.

### Parameters
- **adata** : `AnnData`, optional  
  Pre-loaded AnnData object. If not provided, `data_path` is used.

- **data_path** : `str`, optional  
  Path to `.h5ad` file. Required if `adata` is not provided.

- **cell_type_key** : `str`, optional  
  Key in `.obs` representing cell types. Required if `trained=True`.

- **batch_key** : `str`, optional  
  Key in `.obs` representing batch metadata. Used to create batch index.

- **trained** : `bool`, default: `False`  
  Whether to apply ontology-based training preprocessing.

- **graph** : `networkx.Graph` or `str`, optional  
  Ontology graph or path to `.gml` file used to construct it.

- **highly_variable_genes** : `bool`, default: `True`  
  Whether to filter for highly variable genes during preprocessing.

- **llm_vocab** : `str`, optional  
  Path to GeneFormer vocabulary file.

- **llm_args** : `str`, optional  
  JSON path specifying additional LLM arguments.

---

### 🔧 Internal Attributes
- `adata`: Preprocessed `AnnData` object.
- `ontograph`: OntoGRAPH instance if `trained=True`.
- `batch_index`: Batch indices if `batch_key` is provided.
- `cell_type_index`: List of indices into the ontology vocabulary.
- `llm_vocab`, `llm_args`: LLM configuration and vocabulary for gene models.

---

## 🔍 Methods

### `read_data(highly_variable_genes=True, subset=True)`
Loads and preprocesses an `.h5ad` dataset.

- Normalizes to 1e4 and applies `log1p` transformation if necessary.
- Filters for highly variable genes if `trained=True`.

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
    cell_type_key="cell_type",
    trained=True
)

emb = dataset.get_initial_embeddings(input_dim=50)
hier_labels = dataset.get_cell_type_hierarchy_matrix()
```