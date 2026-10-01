# scFoundation + UniCell Training and Evaluation Pipeline

UniCell adopts a hybrid learning paradigm that integrates the generalization capacity
of single-cell foundation models (scFMs) with the structured biological knowledge of
ontology-guided expert models.

In this framework:
- Foundation models such as **scGPT**, **GeneFormer**, and **scFoundation** are pretrained
  on large-scale single-cell transcriptomic corpora to capture generalizable gene expression
  patterns across tissues, species, and platforms.
- These models encode cells into expressive, low-dimensional embeddings that serve as
  versatile input representations.
- UniCell then uses a Cell Ontology hierarchy together with flat cell-type, tissue, and organism heads to further enhance prediction accuracy.

This notebook demonstrates training and evaluation using the **scFoundation + UniCell** hybrid setup.

> **Version 0.2.1:** `HMCNDataset` converts scFoundation rows to tensors before the trainer and inference loops stack them, fixing the pandas `Series`/tensor mismatch. This repair does not establish end-to-end validation of the full tutorial or reproduction of the paper results. Supply compatible data and pretrained weights and validate the workflow in your environment. The expression, GeneFormer, and scGPT input paths are unchanged.

- 📂 Demo data can be downloaded from:  
https://bgipan.genomics.cn/#/link/Z2z61owwv8WAqbuIcCdB  
🔑 Extraction code: `YC3A`

- 📂 scFoundation checkpoint can be downloaded from:  
https://bgipan.genomics.cn/#/link/2FXZCAUYTDixGJoOmvGH  
🔑 Extraction code: `JXhH`

---

## Workflow Summary
- Load and preprocess single-cell RNA-seq datasets
- Construct and configure training dataset
- Train a UniCell model using a foundation encoder
- Prepare evaluation dataset and continue training or validate
- Predict on test dataset
- Evaluate model performance

---

## Step 1: Load and Preprocess Data

```python
import scanpy as sc

adata_train = sc.read_h5ad("PATH_to_DATA/hsa_liver_train.h5ad")
adata_eval = sc.read_h5ad("PATH_to_DATA/hsa_liver_eval.h5ad")
adata_test = sc.read_h5ad("PATH_to_DATA/hsa_liver_test.h5ad")

adata_train.var_names = adata_train.var["feature_name"].tolist()
adata_eval.var_names = adata_eval.var["feature_name"].tolist()
adata_test.var_names = adata_test.var["feature_name"].tolist()

required_obs = {"cell_type_ontology_term_id", "general_tissue", "organism"}
for name, adata in {"train": adata_train, "eval": adata_eval, "test": adata_test}.items():
    missing = required_obs - set(adata.obs.columns)
    if missing:
        raise KeyError(f"{name} data is missing required obs columns: {sorted(missing)}")
    if adata.obs[list(required_obs)].isna().any().any():
        raise ValueError(f"{name} data contains missing values in required obs columns")
```

Because an in-memory `AnnData` object is supplied below, `scDataset` does not run its file-loading preprocessing. Prepare the matrices according to the scFoundation checkpoint before constructing the datasets.

---

## Step 2: Construct Training Dataset

```python
from unicell.scDataset import scDataset
import json
import pandas as pd

scf_vocab = "PATH_to_scFM/scfoundation/gene_vocab.json"

# UniCell expects a JSON or pickle vocabulary. Build the JSON once from the
# canonical 19,264-gene scFoundation table if it is not already supplied.
gene_table = pd.read_csv(
    "PATH_to_scFM/scfoundation/OS_scRNA_gene_index.19264.tsv",
    sep="\t"
).sort_values("index")
with open(scf_vocab, "w") as handle:
    json.dump(
        dict(zip(gene_table["gene_name"], gene_table["index"].astype(int))),
        handle
    )

sc_train = scDataset(
    adata=adata_train,
    data_path=None,
    cell_type_key="cell_type_ontology_term_id",
    tissue_key="general_tissue",
    species_key="organism",
    trained=True,
    highly_variable_genes=True,
    llm_vocab=scf_vocab,
    llm_args=None
)
```

---

## Step 3: Train UniCell Model

```python
import os
import pickle
from unicell.trainer import UnicellTrainer

input_dim = sc_train.adata.shape[1]
device = "cuda:2"
ckpt_dir = "models/scf_models"
os.makedirs(ckpt_dir, exist_ok=True)

with open(os.path.join(ckpt_dir, 'gene_names.pk'), 'wb') as w1:
    pickle.dump(sc_train.adata.var_names, w1)
sc_train.ontograph.pickle(ckpt_dir)

llm_model_path = "PATH_to_scFM/scfoundation/models.ckpt"

trainer = UnicellTrainer(
    sc_train,
    input_type="scFoundation",
    input_dim=None,
    output_dim=None,
    batch_size=16,
    learning_rate=1e-4,
    num_epochs=10,
    beta=0.1,
    device=device,
    global_layer=128,
    local_layer=64,
    hidden_layer_dropout=0.2,
    ckpt_dir=ckpt_dir,
    llm_model_file=llm_model_path,
    llm_vocab_file=scf_vocab
)
```

---

## Step 4: Prepare Evaluation Data and Continue Training

```python
sc_eval = scDataset(
    adata=adata_eval,
    data_path=None,
    cell_type_key="cell_type_ontology_term_id",
    tissue_key="general_tissue",
    species_key="organism",
    trained=False,
    highly_variable_genes=False,
    llm_vocab=scf_vocab,
    llm_args=None
)
missing_eval_genes = sc_train.adata.var_names.difference(sc_eval.adata.var_names)
if len(missing_eval_genes):
    raise ValueError(f"Evaluation data is missing {len(missing_eval_genes)} training genes")

sc_eval.adata = sc_eval.adata[:, sc_train.adata.var_names]
sc_eval.ontograph = sc_train.ontograph
sc_eval.cell_type_index = sc_eval.get_cell_type_index()

import numpy as np

eval_tissues = sc_eval.adata.obs["general_tissue"].astype(str)
eval_species = sc_eval.adata.obs["organism"].astype(str)
unseen_tissues = sorted(set(eval_tissues) - set(sc_train.tissue_label_dict))
unseen_species = sorted(set(eval_species) - set(sc_train.species_label_dict))
if unseen_tissues or unseen_species:
    raise ValueError(
        f"Labels not seen during training: tissue={unseen_tissues}, organism={unseen_species}"
    )

sc_eval.tissue_label_dict = sc_train.tissue_label_dict
sc_eval.species_label_dict = sc_train.species_label_dict
sc_eval.tissue_index = np.array(
    [sc_train.tissue_label_dict[x] for x in eval_tissues], dtype=np.int64
)
sc_eval.species_index = np.array(
    [sc_train.species_label_dict[x] for x in eval_species], dtype=np.int64
)

trainer.train(scdata_test=sc_eval)
```

---

## Step 5: Predict on Test Dataset

```python
from unicell.anno_predict import unicell_predict
sc_dataset = unicell_predict(
    adata=adata_test,
    filepath=None,
    ckpt_dir=ckpt_dir,
    device=device,
    cell_type_key="cell_type_ontology_term_id",
    tissue_key="general_tissue",
    species_key="organism",
    compute_metrics=True,
    safe_list_path=None
)
```

---

## Step 6: Evaluate Model Predictions

```python
from unicell.evaluate import compute_metrics

cell_type_metrics = compute_metrics(
    scDataset=sc_dataset,
    label_key="cell_type_ontology_term_id",
    prediction_key="predicted_cell_type_ontology_id"
)
tissue_metrics = compute_metrics(
    scDataset=sc_dataset,
    label_key="general_tissue",
    prediction_key="predicted_tissue"
)
organism_metrics = compute_metrics(
    scDataset=sc_dataset,
    label_key="organism",
    prediction_key="predicted_species"
)

print("UniCell cell-type metrics:", cell_type_metrics)
print("UniCell tissue metrics:", tissue_metrics)
print("UniCell organism metrics:", organism_metrics)
```
