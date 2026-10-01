# `OntoGRAPH` Module

This module builds and manipulates a hierarchical ontology graph based on the Cell Ontology (.obo) format. It supports pruning, hierarchy extraction, and label transformation for the cell-type branch of the multi-task model.

---

## 🧬 Class: `OntoGRAPH`

```python
OntoGRAPH(graph=None, cell_type=None)
```

### Description
Initializes an ontology graph from the packaged `cl-basic.obo` resource or an existing NetworkX graph object. To load a different `.obo` path, call `import_ontology(url)` and pass the resulting graph. If `cell_type` is provided, the constructor filters the graph to include relevant nodes and their hierarchical relationships.

Tissue and organism labels are auxiliary flat classification tasks and are not inserted into this Cell Ontology graph. Their class mappings are managed by `scDataset` and `UnicellTrainer`.

### Parameters
- **graph** : `networkx.DiGraph`, optional  
  Pre-loaded ontology graph. If `None`, loads `cl-basic.obo` from the local path.

- **cell_type** : `list[str]`, optional  
  A list of cell type ontology IDs (e.g., `["CL:0000540"]`) used to filter and prune the ontology. Also used to compute the lowest common ancestor and reduce graph size.

---

### 🔧 Internal Attributes
- `graph`: The final subgraph of the ontology.
- `id2name`: Dictionary mapping term IDs to human-readable names.
- `vocab`: Ordered dictionary mapping cell type IDs to vocabulary indices.
- `hierarchical_levels`: Dict of term ID → depth level.
- `hierarchical_array`: Per-level binary ancestor-membership matrices; rows can be multi-hot in a directed acyclic ontology graph.
- `cell_type_idx_set`: Mapping from global → local → vocab index.

---

## 🔍 Methods

### `import_ontology(url=None)`
Reads an OBO-formatted file and constructs a directed ontology graph.

- **url** : `str`, optional  
  Path to `.obo` file. If `None`, defaults to `cl-basic.obo` in the script directory.

Returns: `networkx.MultiDiGraph`

---

### `get_obsolete_id_mapper()`
Returns a dictionary of obsolete ontology term IDs that have been replaced.

Returns: `dict`

---

### `get_id_mapper()`
Returns a dictionary mapping term IDs to names from the graph.

Returns: `dict`

---

### `convert_to_vocab()`
Creates a vocabulary dictionary mapping sorted term IDs to consecutive integer indices.

Returns: `OrderedDict`

---

### `get_siblings(term_id)`
Finds sibling terms under the first direct parent. The implementation assumes that the term has at least one parent.

Returns: `list[str]`

---

### `get_parents(term_id)`
Returns the direct parent(s) of a given term.

Returns: `list[str]`

---

### `get_children(term_id)`
Returns direct children of a given term.

Returns: `list[str]`

---

### `get_all_ancestors(node, inclusive=False)`
Finds all ancestor terms of a given node.

Returns: `list[str]`

---

### `get_all_descendants(nodes, inclusive=True)`
Finds all descendant terms of a node. The current implementation expects `nodes` to be a single term ID string.

Returns: `list[str]`

---

### `get_hierarchical_levels()`
Computes the hierarchy depth for each node starting from the root.

Returns: `dict[str, int]`

---

### `get_hierarchical_array()`
Constructs a list of binary matrices (torch tensors) representing the hierarchical label structure.

Returns: `list[torch.Tensor]`

---

### `get_node_hclasses()`
Generates a mapping from node ID to local class index per hierarchy level.

Returns: `dict[int, dict[str, int]]`

---

### `get_cell_type_idx_set()`
Builds a mapping of hierarchy level and class index to global vocabulary index.

Returns: `dict[int, dict[int, int]]`

---

### `pickle(output)`
Saves the ontology graph and object state to disk.

- **output** : `str`  
  Path to the output directory.

Creates:
- `ontoGraph.graph.gml`
- `ontoGraph.pk`

After writing the GML file, this method sets the in-memory instance's `graph` attribute to `None` before pickling it. Reload the graph with `load_ontograph(output)` before performing further graph traversal on that saved object.

---

## ✅ Example

```python
from unicell.ontoGRAPH import OntoGRAPH

og = OntoGRAPH(cell_type=["CL:0000540", "CL:0000236"])

print(og.hierarchical_levels)
print(og.vocab)
```
