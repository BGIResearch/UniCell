# UniCell Fig5 0.2.1 wheel release

Post-installation usage guides: [English](USAGE.md) | [简体中文](USAGE_ZH.md).

This packaging layout keeps the UniCell application and model independent from
architecture-specific scientific and CUDA wheels.

## Supported targets

- Linux aarch64, CPython 3.9, the cluster-provided PyTorch 2.0.0 CUDA 11.6 build.
- Linux x86_64, CPython 3.9, PyTorch 2.0.0 CUDA 11.7 as the initial compatibility target.

The `unicell_fig5-0.2.1-py3-none-any.whl` application wheel and model files are
shared. Each target has its own requirements file and wheelhouse.

## Build a release on the current ARM cluster

The bootstrap bundle requires files that are not stored in Git. Before running
the release builder, prepare:

- A Python 3.9 interpreter with `build`, `setuptools>=68`, and `wheel` installed.
  Set `UNICELL_BUILD_PYTHON` to that interpreter if `python3` is a different version.
- The seven checkpoint files in `models/last_version_gml/`:
  `unicell_v1.best.pth`, `gene_names.pk`, `ontoGraph.pk`,
  `ontoGraph.graph.gml`, `celltype_dict.pk`, `tissue_dict.pk`, and
  `species_dict.pk`. Obtain them from the versioned expression checkpoint
  archive linked in the main repository README and verify its checksum.
- An existing directory containing the two cluster-specific ARM wheels:
  `torch-2.0.0+cuda11.6.gcc9.3-cp39-cp39-linux_aarch64.whl` and
  `torchtext-0.15.2a0+4571036-cp39-cp39-linux_aarch64.whl`.
  Set `UNICELL_ARM_TORCH_WHEEL_DIR` to that directory. The builder does not
  download these wheels.
- A release output location with no existing `unicell_fig5-0.2.1/` directory
  or corresponding bootstrap archive. Set `UNICELL_RELEASE_OUT` to choose a
  different output directory when rebuilding.

```bash
git clone https://github.com/luyi98/unicell.git
cd unicell
python3.9 -m pip install build 'setuptools>=68' wheel
# Place the checkpoint files in models/last_version_gml/ before continuing.
UNICELL_BUILD_PYTHON=/absolute/path/to/python3.9 \
UNICELL_ARM_TORCH_WHEEL_DIR=/absolute/path/to/arm-wheels \
  bash packaging/scripts/build_release.sh
```

To build only the application wheel, use
`UNICELL_BUILD_PYTHON=/absolute/path/to/python3.9 bash packaging/scripts/build_wheel.sh`;
checkpoint files and ARM dependency wheels are not required for that step.

Outputs:

```text
dist/unicell_fig5-0.2.1-py3-none-any.whl
dist/releases/unicell_fig5-0.2.1/
dist/releases/unicell_fig5-0.2.1-multiarch-bootstrap.tar.gz
```

The bootstrap archive includes the ARM PyTorch and TorchText wheels, but not all
third-party dependency wheels. To make a platform fully offline, run the
wheelhouse population script with a Python 3.9 interpreter on that native
architecture:

```bash
UNICELL_WHEEL_PYTHON=/path/to/python3.9 \
  bash populate_wheelhouse.sh /path/to/unicell_fig5-0.2.1
```

Run this once on ARM and once on x86_64. Native builds are required for runtime
testing even when wheels are downloaded using cross-platform pip options.

## Install without root

The installer creates an isolated environment owned by the current user:

```bash
UNICELL_ENV_MANAGER=/absolute/path/to/mamba \
  bash install.sh /absolute/path/to/unicell_runtime
```

If Python 3.9 is already available, Conda/Mamba is not required:

```bash
UNICELL_BASE_PYTHON=/absolute/path/to/python3.9 \
  bash install.sh /absolute/path/to/unicell_runtime
```

For a populated offline wheelhouse:

```bash
UNICELL_OFFLINE=1 \
UNICELL_BASE_PYTHON=/absolute/path/to/python3.9 \
  bash install.sh /absolute/path/to/unicell_runtime
```

The installer detects `aarch64` or `x86_64`, installs the corresponding locked
dependencies, runs `pip check`, imports all entry points, and validates the
bundled checkpoint structure without allocating the full model. Run
`smoke_test.py --instantiate-model` separately when at least about 1 GB of extra
RAM is available.

## Run prediction

```bash
source /absolute/path/to/unicell_runtime/bin/activate

unicell-predict \
  --input /absolute/path/to/input.h5ad \
  --ckpt_dir /absolute/path/to/release/common/models/last_version_gml \
  --output /absolute/path/to/output.h5ad \
  --safe_list /absolute/path/to/release/common/safe_list.json \
  --device cuda \
  --batch_size 2048
```

## Train with a GeneFormer or scGPT encoder

The wheel installs `unicell-train-backbone` and lightweight vocabulary
resources, but does not redistribute GeneFormer or scGPT model weights. Supply
foundation-model assets by path and validate them before training:

```bash
unicell-train-backbone \
  --input_type GeneFormer \
  --train_h5ad /data/train.h5ad \
  --eval_h5ad /data/valid.h5ad \
  --llm_model_file /models/Geneformer-V1-10M \
  --ckpt_dir /output/unicell_geneformer \
  --validate_only

unicell-train-backbone \
  --input_type scGPT \
  --train_h5ad /data/train.h5ad \
  --eval_h5ad /data/valid.h5ad \
  --llm_model_file /models/scgpt/best_model.pt \
  --llm_vocab_file /models/scgpt/vocab.json \
  --llm_args_file /models/scgpt/args.json \
  --ckpt_dir /output/unicell_scgpt \
  --validate_only
```

Remove `--validate_only` to start training. GeneFormer uses the vocabulary
bundled in the wheel by default; scGPT requires the vocabulary and `args.json`
from the same pretrained release as its `.pt` file. Newly trained checkpoints
store lightweight reconstruction assets in `CKPT_DIR/backbone/` and all trained
weights in `unicell_v1.best.pth`. Move the entire checkpoint directory; portable
inference then needs only `--ckpt_dir`, not the original foundation checkpoint.
The bundle contains `manifest.json` plus `config.json`/`gene_vocab.json` for
GeneFormer or `args.json`/`vocab.json` for scGPT; it contains no model weights.

## Compatibility boundary

The model checkpoint is a PyTorch state dictionary and is expected to be shared
between aarch64 and x86_64. CUDA, PyTorch, TorchText, NumPy, SciPy, H5py, Numba,
and similar compiled packages are not shared. The x86_64 dependency set must pass
the bundled smoke test and a small end-to-end H5AD inference before production use.
The populated ARM wheelhouse was tested on aarch64 with glibc 2.28 and the
cluster CUDA 11.6 runtime; another ARM machine must provide compatible system
CUDA/driver libraries. `flash_attn` is an optional acceleration and its absence
does not block the bundled model path. The vendored scGPT `eval_scib_metrics`
helper additionally requires `scib==1.1.4`, but UniCell prediction does not.

## Licenses

Original UniCell contributions use the MIT License. Incorporated code and data
retain their own licenses, including GPL v3; see `LICENSE`,
`THIRD_PARTY_NOTICES.md`, and `licenses/` in the assembled release. These files
are also included with the application wheel and source distribution. Model
weights and datasets require their own applicable permissions. See the notices
for the remaining SCAD/DSBN documentation gaps before public redistribution.
