# UniCell Command-line Interfaces

The `unicell-fig5` 0.2.1 wheel exposes four console commands for prediction and training. The wheel requires Python 3.9 and contains application code and ontology resources, but not the trained model weights.

---

## 📦 Checkpoint Contract

Prediction requires one consistent checkpoint directory containing:

```text
unicell_v1.best.pth
gene_names.pk
ontoGraph.pk
ontoGraph.graph.gml
celltype_dict.pk
tissue_dict.pk
species_dict.pk
```

Do not mix files from different training runs. The current `models/last_version_gml` checkpoint uses 60,695 input features, 718 cell-type labels, 60 tissue labels, and 2 organism labels. These dimensions describe that checkpoint rather than a fixed limit of the wheel.

These seven files are sufficient for an expression-input checkpoint. Portable GeneFormer/scGPT checkpoints created by `unicell-train-backbone` additionally include lightweight configuration and vocabulary files under `backbone/`; the trained backbone parameters remain in `unicell_v1.best.pth`. Legacy checkpoints may still require the original external paths or the `--backbone_model`, `--backbone_vocab`, and `--backbone_args` overrides.

---

## 🧬 `unicell-predict`

Runs single-file inference and writes an annotated `.h5ad`:

```bash
unicell-predict \
  --input /absolute/path/to/input_log1p.h5ad \
  --ckpt_dir /absolute/path/to/models/last_version_gml \
  --output /absolute/path/to/output_annotated.h5ad \
  --device cuda \
  --batch_size 512 \
  --cell_type_key cell_type_ontology_term_id \
  --tissue_key general_tissue \
  --species_key organism \
  --save_metrics_json
```

`--input` and `--ckpt_dir` are required. When `--output` is omitted, the command writes `<input>_annotated.h5ad`; the safe list is optional. When `--device` is omitted, the command selects CUDA when available and otherwise uses CPU.

The command applies the training/evaluation preprocessing heuristic to a copy of the input: for nonnegative matrices with more than 1,000 genes it filters cells with fewer than 200 detected genes, and values with a maximum greater than 25 are log-transformed after optional total-count normalization for integer-like counts. Omitting `--safe_list` performs unconstrained decoding.

If ground-truth columns are available, the command prints accuracy and macro F1 for cell type, tissue, and organism. `--save_metrics_json` controls JSON output; it is not required for printing metrics.

See [`unicell_predict`](unicell_predict.md) for the complete returned fields.

---

## 🧩 `unicell-predict-chunks`

Runs inference on a large `.h5ad` one `obs` group at a time:

```bash
unicell-predict-chunks \
  --input /absolute/path/to/large_input_log1p.h5ad \
  --ckpt_dir /absolute/path/to/models/last_version_gml \
  --out_dir /absolute/path/to/prediction_chunks \
  --sample_key sample \
  --device cuda \
  --batch_size 512 \
  --compression gzip \
  --compression_opts 4 \
  --skip_done \
  --save_metrics_json
```

The input is opened in backed read-only mode and each value of `obs[sample_key]` is loaded into memory separately. Every completed chunk produces an `.h5ad` and a CSV containing the observation name, cell-type prediction, tissue prediction, organism/species prediction (`predicted_species`), and sample value. A final `chunk_summary.csv` records completion status and optional metrics. Unlike the single-file command, this command falls back to CPU when CUDA is unavailable.

---

## 🧠 `unicell-train`

Runs the fixed training workflow packaged in the wheel:

```bash
unicell-train \
  --train_h5ad /absolute/path/to/train.h5ad \
  --eval_h5ad /absolute/path/to/valid.h5ad
```

Both files must contain non-missing values in:

```text
obs["cell_type_ontology_term_id"]
obs["general_tissue"]
obs["organism"]
```

The validation file must contain every training gene and cannot contain tissue or organism labels unseen during training. Validation expression columns are reordered to the training gene order, and its tissue and organism labels reuse the training dictionaries.

The wheel command exposes only the two data paths. Hyperparameters, label keys, and the `models/checkpoints` output directory are fixed in `train_Unicell.py`. The repository-only `train_epoch_unicell.py` implements the final seed-rotating experiment workflow but is not installed as a console command.

---

## 🧠 `unicell-train-backbone`

Runs the configurable 0.2.1 training workflow for expression, GeneFormer, or
scGPT input. It validates data and backbone assets before model construction:

```bash
unicell-train-backbone \
  --input_type GeneFormer \
  --train_h5ad /absolute/path/to/train.h5ad \
  --eval_h5ad /absolute/path/to/valid.h5ad \
  --llm_model_file /absolute/path/to/Geneformer-V1-10M \
  --ckpt_dir /absolute/path/to/output_checkpoint \
  --validate_only
```

Remove `--validate_only` to start training. For scGPT, also provide the matching
`--llm_vocab_file` and `--llm_args_file`. The resulting portable checkpoint
stores lightweight reconstruction assets next to the UniCell checkpoint.
