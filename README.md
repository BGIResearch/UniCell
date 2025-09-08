# UniCell: Towards a Unified Solution for Cell Annotation, Nomenclature Harmonization, Atlas Construction in Single-Cell Transcriptomics

UniCell is a hierarchical deep learning framework that integrates Cell Ontology with transcriptomic features to enable accurate, scalable, and cross-species cell type annotation, nomenclature harmonization, and atlas-level integration in single-cell transcriptomics.

## 📦 Features

- 🧠 **Ontology-aware prediction**  
  Leverages Cell Ontology to ensure consistent and interpretable annotations across hierarchical levels.

- 🧬 **Hierarchical multi-task learning**  
  Simultaneously trains local and global classifiers to enhance accuracy and granularity.

- 🌍 **Cross-species and cross-tissue harmonization**  
  Embeds cells from diverse atlases into a shared latent space for robust integration.

- 🔍 **Rare and novel cell detection**  
  Identifies unseen or low-abundance populations using confidence-guided novelty detection.

- 🧱 **Foundation model enhancement**  
  Boosts the accuracy and structure of single-cell foundation models via hierarchical supervision.

---

## 📁 Repository Structure

```
unicell/
├── anno_predict.py     # Prediction pipeline for UniCell
├── dataset.py          # Data loaders and preprocessing
├── evaluate.py         # Benchmarking and evaluation routines
├── hmcn.py             # Hierarchical classification network (HMCN)
├── loss.py             # Loss functions for training
├── ontoGRAPH.py        # Ontology graph construction and utilities
├── scDataset.py        # Dataset handling with ontology support
├── trainer.py          # Training loop and optimization
├── utils/              # General utility functions
├── cl-basic.obo        # Cell ontology in OBO format
├── graph.gml           # Ontology graph structure in GML
├── repo/               # Pretrained single-cell foundation models
└── __init__.py         # Module init file
```

---

## 🚀 Getting Started

### 📚 Installation

```bash
git clone https://github.com/huluni/unicell.git
cd unicell
pip install -r requirements.txt
```

> Ensure your Python environment includes `scanpy`, `pytorch`, `networkx`, `matplotlib`, and `seaborn`.

---

## 🧪 Example Usage

```python
from unicell.anno_predict import unicell_predict
import scanpy as sc

adata = sc.read_h5ad("your_input_file.h5ad")
result = unicell_predict(adata=adata, ckpt_dir="path/to/checkpoint", device="cuda")
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

UniCell is developed by **BGI Research**. If you use this tool in your research, please cite the corresponding publication:

Hu et al., UniCell: Towards a Unified Solution for Cell Annotation, Nomenclature Harmonization, Atlas Construction in Single-Cell Transcriptomics, bioRxiv, 2025.
https://doi.org/10.1101/2025.05.06.652331