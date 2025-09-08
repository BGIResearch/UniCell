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
- UniCell then uses expert models informed by biological ontologies to further enhance prediction accuracy.

This notebook demonstrates training and evaluation using the **scFoundation + UniCell** hybrid setup.

- 📂 Demo data can be downloaded from:  
https://bgipan.genomics.cn/#/link/Z2z61owwv8WAqbuIcCdB  
🔑 Extraction code: `YC3A`

- 📂 GeneFormer checkpoint can be downloaded from:  
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
```

---

## Step 2: Construct Training Dataset

```python
from unicell.scDataset import scDataset

scf_vocab = "PATH_to_scFM/scfoundation/gene_vocab.json"
sc_train = scDataset(
    adata=adata_train,
    data_path=None,
    cell_type_key="cell_type_ontology_term_id",
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
    input_type="GeneFormer",
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
    trained=False,
    highly_variable_genes=False,
    llm_vocab=scf_vocab,
    llm_args=None
)
sc_eval.adata = sc_eval.adata[:, sc_train.adata.var_names]
sc_eval.ontograph = sc_train.ontograph
sc_eval.cell_type_index = sc_eval.get_cell_type_index()

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
    device=device
)
```

---

## Step 6: Evaluate Model Predictions

```python
from unicell.evaluate import compute_metrics

eval_metrics = compute_metrics(
    sc_dataset.adata.obs['cell_type_ontology_term_id'],
    sc_dataset.adata.obs["predicted_cell_type_ontology_id"]
)
print("CATree eval metrics:", eval_metrics)
```
