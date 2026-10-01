# 📘 Novel Cell Type Detection with UniCell

This tutorial demonstrates how to use **UniCell** scores to flag candidate novel or unseen cell populations using a pre-trained model and reference ontology. The current classifier predicts only cell types seen during training; novelty is identified downstream from uncertainty in the known-class logits and then validated with clustering and marker genes.

---

## 📂 Dataset and Setup
- Demo data can be downloaded from:  
https://bgipan.genomics.cn/#/link/15r5PAxuJO8NXaiGFIv3  
🔑 Extraction code: `OSxv`

- Pretrained models can be downloaded from:  
https://figshare.com/articles/online_resource/02_normal_to_disease/28901045

Make sure to download the dataset folder from your data source. The following assumes a structure like:

- `data/normal.h5ad`
- `data/disease.h5ad`
- `models/last_version_gml`

The checkpoint must follow the current seven-file contract and include the tissue and species heads; see the [command-line reference](../API/command_line.md). The embedded plots are retained as workflow illustrations and should be regenerated with the current joint checkpoint before reporting numerical results.

---

## 🧭 Workflow Summary

1. Prepare the reference dataset and the query dataset.
2. Detect novel cell types.
3. Characterize novel cell types.


---

```python
import scanpy as sc
import pickle
from unicell.anno_predict import unicell_predict
from unicell.scDataset import scDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
```

## Step 1: Prepare the reference dataset and the query dataset


```python
#### Prepare the training set and the evaluation set
# unicell_predict copies this AnnData and applies the training/evaluation
# expression preprocessing heuristic.
adata_test = sc.read_h5ad("data/disease.h5ad")
ckpt_dir = "models/last_version_gml"

#### prepare sc_dataset for following analysis
sc_dataset = unicell_predict(
    adata=adata_test,
    batch_size=512,
    filepath=None,
    ckpt_dir=ckpt_dir,
    device="cuda",
    cell_type_key="cell_type_ontology_term_id",
    tissue_key="tissue",
    species_key="organism",
    compute_metrics=True,
    safe_list_path=None
)

sc_dataset.adata.obs[
    ["predicted_cell_type_ontology_id", "predicted_tissue", "predicted_species"]
].head()
```

The tissue and organism predictions provide context checks for the candidate population. The novelty score below uses only `cls_emb`, which contains raw logits over the cell types seen during training.

This downloaded demo uses `adata.obs["tissue"]`, so the call overrides `tissue_key`. The project-wide default for other datasets is `adata.obs["general_tissue"]`.

## Step 2: Detect novel cell types


```python
adata_normal = scDataset(
    data_path="data/normal.h5ad",
    trained=False,
    highly_variable_genes=False
).adata
adata_normal.var_names = adata_normal.var["feature_name"].tolist()
adata = sc.concat([sc_dataset.adata, adata_normal])

sc.pp.scale(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_pcs = 20)
sc.tl.umap(adata)

adata_q = adata[adata.obs["disease"].isin(["follicular lymphoma"])].copy() # query adata
adata_r = adata[adata.obs["disease"].isin(["normal"])].copy() # reference adata

sc.pp.neighbors(adata_q, n_pcs=20)
sc.tl.leiden(adata_q)
```


```python
sc.pl.umap(adata_q, color = ["leiden"], legend_loc = "on data")
```
![png](02_umap_leiden.png)
    



```python
import torch
cls_emb = torch.tensor(sc_dataset.adata.obsm["cls_emb"])
# Convert known-cell-type logits into probabilities before scoring uncertainty.
data = torch.softmax(cls_emb, dim=1).numpy()

adata_q.obs["cell_type_var"] = np.var(data, axis = 1)
sc.pl.violin(adata_q, keys=["cell_type_var"], groupby="leiden")
```
![png](02_violin_leiden.png)
    



```python
# Group data by 'leiden' clustering and get the groupings
grouper = adata_q.obs.groupby(["leiden"]).grouper

# Compute the median of 'cell_type_var' for each 'leiden' cluster
group_var_means = adata_q.obs["cell_type_var"].groupby(grouper).median()

# Import median absolute deviation (not used in this code block but commonly used for robust statistics)
from scipy.stats import median_abs_deviation

# Compute standard deviation of median values across groups
sd = np.std(group_var_means)

# Calculate the first and third quartiles (Q1 and Q3)
Q1 = np.percentile(group_var_means, 25)
Q3 = np.percentile(group_var_means, 75)
IQR = Q3 - Q1  # Interquartile range

# Define the lower and upper bounds for outlier detection using IQR method
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Set threshold to the lower bound for identifying low-value outliers
threshold = lower_bound

# Identify groups whose median values fall below the lower threshold
outliers = group_var_means[group_var_means < threshold]

# Convert group_var_means to a pandas Series (ensure it's the correct type)
group_var_means = pd.Series(group_var_means)

# Create a DataFrame for plotting
data = pd.DataFrame({'Value': group_var_means.values})

# Plot a boxplot to show distribution of group median values
plt.figure(figsize=(3, 4))
sns.boxplot(y='Value', data=data, fill=None)
plt.ylabel('Value')

# Overlay all values as transparent black dots
plt.scatter(np.zeros(len(group_var_means)), group_var_means.values, color='black', label='Values', alpha=0.2)

# Highlight outlier values in red
plt.scatter(np.zeros(len(outliers)), outliers.values, color='red', label='Outliers', zorder=5)

# Add legend to distinguish points
plt.legend()

# Improve layout
plt.tight_layout()
plt.show()
```
![png](02_boxplot_leiden.png)
    



```python
#### infer cell type UMAP
adata_test = sc_dataset.adata.copy()
adata_test.obsm["X_umap"] = adata[adata_test.obs_names].obsm["X_umap"]
adata_test.obs["leiden"] = adata_q.obs["leiden"]

adata_test.obs["predicted_cell_type_name"] = [
    sc_dataset.ontograph.id2name.get(cell_type_id, cell_type_id)
    for cell_type_id in adata_test.obs["predicted_cell_type_ontology_id"]
]
adata_test.obs["predicted_cell_type_name"] = pd.Categorical(
    adata_test.obs["predicted_cell_type_name"]
)
adata_test.obs["predicted_cell_type_checked"] = adata_test.obs[
    "predicted_cell_type_name"
].astype(str)
adata_test.obs.loc[
    adata_test.obs["leiden"].isin(["22"]), "predicted_cell_type_checked"
] = "unknown"
adata_test.obs["predicted_cell_type_checked"] = adata_test.obs["predicted_cell_type_checked"].astype("category")
checked_categories = list(dict.fromkeys(
    adata_test.obs["cell_type"].astype(str).tolist()
    + adata_test.obs["predicted_cell_type_checked"].astype(str).tolist()
    + ["unknown"]
))
adata_test.obs["predicted_cell_type_checked"] = adata_test.obs[
    "predicted_cell_type_checked"
].cat.set_categories(checked_categories)

sc.pl.umap(adata_test, color = ["cell_type", "predicted_cell_type_name", "predicted_cell_type_checked"], size=50, show=False)
plt.show()
```
![png](02_umap_refined.png)
    


## Step 3: Characterize novel cell types


```python
sc.tl.rank_genes_groups(adata_test, groupby="predicted_cell_type_checked", method="wilcoxon", use_raw=False)
dedf = sc.get.rank_genes_groups_df(adata_test, group=None)
marker_dict = dedf.groupby("group").apply(
    lambda x: x.nlargest(10, "scores")["names"].tolist()
).to_dict()
adata_test.obs["predicted_cell_type_checked"] = adata_test.obs["predicted_cell_type_checked"].cat.set_categories(list(adata_test.obs["predicted_cell_type_checked"].cat.categories)[::-1])
sc.pl.dotplot(adata_test, groupby="predicted_cell_type_checked", var_names=marker_dict["unknown"], use_raw=False, vmax=5, swap_axes=True, show=False)
plt.show()
```
![png](02_dotplot_refined.png)

```python
import os
import torch
cls_emb = torch.tensor(sc_dataset.adata.obsm["cls_emb"])
data = torch.softmax(cls_emb, dim=1).numpy()
data = data[adata_test.obs["predicted_cell_type_checked"].isin(["unknown"])]

with open(os.path.join(ckpt_dir, 'celltype_dict.pk'), 'rb') as f:
    label_dict = pickle.load(f)
    idx2label = {label_dict[label]: label for label in label_dict}

cell_type_probs = pd.DataFrame(data)
cell_type_probs.columns = [sc_dataset.ontograph.id2name[v] for k, v in idx2label.items()]

#### Calculate the mean of each column and sort the columns
sorted_cell_type_probs = cell_type_probs.mean().sort_values(ascending=False)

#### Reorder the DataFrame based on the sorted column order
sorted_cell_type_probs_df = cell_type_probs[sorted_cell_type_probs.index]

#### Now you can proceed to plot the violin plot with the sorted DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

#### Create a horizontal violin plot
palette = dict(zip(
    sorted_cell_type_probs_df.columns,
    sns.color_palette("husl", n_colors=sorted_cell_type_probs_df.shape[1])
))
sns.violinplot(data=sorted_cell_type_probs_df, orient='h', palette=palette)

#### Add labels and title
plt.ylabel('Cell Types')
plt.xlabel('Probability')
plt.title('Distribution of Cell Type Probabilities (Sorted)')

#### Show the plot
plt.tight_layout()
plt.show()
```


    
![png](02_violin_refined.png)
    

