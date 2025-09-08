
# 🧠 UniCell Model Architecture

This module defines the **HMCN (Hierarchical Multi-Label Classification Network)** architecture used in UniCell for hierarchical cell-type classification.

---

## 📦 Components

### 1. `Encoder`

A feed-forward neural encoder designed to process representations from different input types (`expr`, `scGPT`, etc.).

```python
Encoder(input_type, d_model, output_dim=128, dropout=0.1)
```

- `input_type`: `"expr"` or `"scGPT"` influences dropout behavior.
- Applies a multi-layer perceptron with normalization and ReLU activations.

---

### 2. `ClsDecoder`

A decoder used for classification tasks. Applies multiple linear layers with activations and layer normalization before producing final logits.

```python
ClsDecoder(d_model, n_cls, nlayers=3)
```

- `d_model`: Input embedding dimension
- `n_cls`: Number of output classes

---

### 3. `HMCN` – Hierarchical Multi-Label Classifier

The core architecture of UniCell for hierarchical classification. Integrates LLM-based embeddings and feeds into multi-depth local and global layers.

```python
HMCN(input_type, input_dim, output_dim, num_classes, hierarchical_depth, ...)
```

- Supports input types: `"scFoundation"`, `"GeneFormer"`, `"scGPT"`
- Freezes selected LLM layers for efficiency
- Builds multiple global and local classifiers per hierarchical level
- Produces:
  - Global representation
  - Final class prediction
  - Per-level predictions
  - Global classification logits

---

## ⚙️ LLM Integration

- **scFoundation**: Loaded via `load_model_frommmf`, uses transformer-based encoder with token & position embeddings.
- **GeneFormer**: `BertForMaskedLM` (via `transformers`)
- **scGPT**: Custom `TransformerModel` with flexible configurations loaded via:

```python
load_gpt_model(vocab, args, model_file)
```

---

## 🧊 Freezing Model Layers

The helper function `freezon_model` allows freezing model parameters except for specified layers.

```python
freezon_model(model, keep_layers=["encoder.layer.11", ...])
```

---

## 🔄 Forward Pass Flow

The model supports various input types. During the forward pass:

1. LLM encoder (e.g. GeneFormer or scGPT) produces embeddings.
2. `Encoder` refines the representation.
3. Each hierarchical level passes through:
   - Global transformation layer
   - Local classifier
4. Final prediction from:
   - Global linear classifier
   - Hierarchy-level outputs
   - Global classification layer

---

## 📌 Notes

- Hierarchical levels are dynamically constructed based on `hierarchical_depth`.
- Local classifiers output binary/multi-label probabilities.
- Final classification uses sigmoid (for multi-label) or raw logits depending on input type.

---
