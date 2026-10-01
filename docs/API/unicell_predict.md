# `unicell_predict` Function

This module provides the inference interface for joint cell-type, tissue, and organism prediction using a trained UniCell model. It loads the saved model checkpoint, aligns input genes, and annotates cells with ontology paths, three task outputs, embeddings, and classifier logits.

---

## 🧬 Function: `unicell_predict`

```python
unicell_predict(
    adata,
    filepath,
    ckpt_dir=None,
    batch_size=512,
    device=None,
    cell_type_key="cell_type_ontology_term_id",
    tissue_key="general_tissue",
    species_key="organism",
    compute_metrics=True,
    safe_list_path=None,
    backbone_model_file=None,
    backbone_vocab_file=None,
    backbone_args_file=None,
    initialize_backbone_from_config=None
)
```

### Description
Loads a trained UniCell model and predicts Cell Ontology IDs, tissues, and organisms for new single-cell data. In code, the organism task is named `species`.

### Parameters
- **adata** : `AnnData`  
  Single-cell expression data to annotate. Inference copies the object and applies the same heuristic preprocessing used by the training/evaluation reader: nonnegative matrices with more than 1,000 genes are cell-filtered, and values with a maximum greater than 25 are log-transformed after optional total-count normalization for integer-like counts.

- **filepath** : `str`, optional  
  Path associated with the input dataset. It may be `None`; when `adata` is supplied, inference uses the in-memory object and retains this value only on the dataset wrapper.

- **ckpt_dir** : `str`  
  Required in practice. Directory containing `unicell_v1.best.pth`, `gene_names.pk`, `ontoGraph.pk`, `ontoGraph.graph.gml`, `celltype_dict.pk`, `tissue_dict.pk`, and `species_dict.pk`.

Portable GeneFormer/scGPT checkpoints created by `unicell-train-backbone` include the lightweight configuration and vocabulary needed to reconstruct the backbone before loading trained weights from `unicell_v1.best.pth`. Legacy checkpoints may still require the original external paths or explicit override arguments.

- **batch_size** : `int`, default: `512`  
  Mini-batch size for inference.

- **device** : `str`, optional  
  Target computation device, such as `"cuda"` or `"cpu"`.

- **cell_type_key** : `str`, default: `"cell_type_ontology_term_id"`  
  Ground-truth Cell Ontology ID column. It does not condition prediction and is used for metrics when requested and present.

- **tissue_key** : `str`, default: `"general_tissue"`  
  Ground-truth tissue column. If present it is indexed on the dataset wrapper, but it does not condition prediction; it is used for metrics when requested.

- **species_key** : `str`, default: `"organism"`  
  Ground-truth organism column. If present it is indexed on the dataset wrapper, but it does not condition prediction; it is used for metrics when requested.

- **compute_metrics** : `bool`, default: `True`  
  Whether to print accuracy and macro F1 independently for each task with an available ground-truth column.

- **safe_list_path** : `str`, optional  
  Path to a JSON file for constrained decoding. It can provide global candidate lists and the linked constraints `species -> tissue -> cell_type`.

- **backbone_model_file**, **backbone_vocab_file**, **backbone_args_file** : `str`, optional
  Override stale external backbone paths stored by a legacy GeneFormer/scGPT checkpoint.

- **initialize_backbone_from_config** : `bool`, optional
  Construct the backbone from its configuration before loading the complete UniCell state dict. When omitted, portable checkpoint metadata selects the behavior automatically.

### Returns
- **scDataset** : `scDataset`  
  Updated dataset containing:
  - `.obs['predicted_cell_type']`
  - `.obs['predicted_cell_type_ontology_id']`
  - `.obs['predicted_tissue']`
  - `.obs['predicted_species']`
  - `.obs['level_0']`, ..., `.obs['level_k']` for the predicted Cell Ontology path
  - `.obsm['unicell_emb']` for the final HMCN global cell representation
  - `.obsm['cls_emb']`, `.obsm['tissue_cls_emb']`, and `.obsm['species_cls_emb']` for classifier logits
  - `.uns['cls_cell_type']`, `.uns['cls_tissue']`, and `.uns['cls_species']` for the corresponding logit-column labels

In version 0.2.1, both `predicted_cell_type` and `predicted_cell_type_ontology_id` contain Cell Ontology IDs rather than natural-language cell-type names. The flat cell-type head selects among the labels in `celltype_dict.pk`; it does not directly emit unseen ontology nodes. The `level_*` columns are added afterward by tracing the selected ID through the saved ontology graph. The classifier matrices are raw logits, not probabilities.

---

## 🔒 Optional Constrained Decoding

Without a safe list, the cell-type, tissue, and organism heads are decoded independently with `argmax`. A safe-list JSON can restrict candidates globally and by the predicted organism and tissue:

```json
{
  "global": {
    "species": ["Homo sapiens"],
    "tissue": ["blood"],
    "cell_type": ["CL:0000624", "CL:0000236"]
  },
  "hierarchy": {
    "Homo sapiens": {
      "tissue": {
        "blood": {
          "cell_type": ["CL:0000624", "CL:0000236"]
        }
      }
    }
  }
}
```

Decoding proceeds as `species -> tissue -> cell_type`, intersecting global and linked candidate sets. Names must exactly match the checkpoint dictionaries and cell types must use Cell Ontology IDs. Unknown entries produce warnings; a missing or empty allowed set falls back to unconstrained `argmax` for that step.

---

## 🔍 Helper Function: `read_data`

```python
read_data(
    adata,
    filepath,
    ckpt_dir,
    llm_vocab,
    llm_args,
    tissue_dict=None,
    species_dict=None,
    tissue_key="general_tissue",
    species_key="organism"
)
```

### Description
Prepares input data by loading the saved ontology graph and aligning expression columns to the exact order in `gene_names.pk`. Extra input genes are ignored and missing checkpoint genes are filled with zero. At least one input gene must match.

The returned `AnnData` is rebuilt around the aligned matrix and retains `obs`, `obsm`, and `uns`; it does not preserve every original `var`, layer, raw, `varm`, or `obsp` field. Keep the original input file if those structures are needed.

### Returns
- `scDataset`: Gene-aligned and ontology-linked dataset.

---

## 🔍 Helper Function: `collate_fn_with_args`

```python
collate_fn_with_args(input_type, model, dataset)
```

### Description
Custom collation logic to format batch data based on the input type.

- For `"GeneFormer"` input, pads token sequences.
- For `"expr"` input, loads expression rows in one batch operation.
- It preserves tissue and organism fields in the five- or six-item dataset batch.

Returns:
- `callable`: DataLoader-compatible collate function

---

## ✅ Example

```python
from unicell.anno_predict import unicell_predict
import anndata as ad

# Raw nonnegative integer counts are normalized/log-transformed automatically
# when they trigger the inference preprocessing heuristic.
adata = ad.read_h5ad("example_input.h5ad")
annotated_dataset = unicell_predict(
    adata=adata,
    filepath="example_input.h5ad",
    ckpt_dir="./checkpoints",
    device="cuda",
    cell_type_key="cell_type_ontology_term_id",
    tissue_key="general_tissue",
    species_key="organism",
    compute_metrics=True,
    safe_list_path=None
)

annotated_dataset.adata.obs[
    ["predicted_cell_type_ontology_id", "predicted_tissue", "predicted_species"]
].head()
```

The ground-truth columns are optional for inference. If any key is absent, metrics for that task are skipped while all three predictions are still produced.
