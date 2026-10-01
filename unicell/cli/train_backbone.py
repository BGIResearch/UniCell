#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Installed UniCell CLI for training with a configurable encoder backbone.

This is an additive entry point which preserves the original
``train_Unicell.py`` command while exposing configurable training through the
installed wheel.

Supported input types:
    - expr
    - GeneFormer
    - scGPT

The startup checks in this file run automatically before any dataset is fully
loaded or a model is initialized. Use ``--validate_only`` to run the same checks
without starting training.
"""

from __future__ import annotations

import argparse
from importlib import resources
import json
import math
import os
import pickle
import random
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple


import anndata as ad
import numpy as np
import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from unicell.scDataset import scDataset


SUPPORTED_INPUT_TYPES = ("expr", "GeneFormer", "scGPT")


def _bundled_geneformer_vocab() -> Path:
    """Return the installed GeneFormer vocabulary as a real filesystem path."""
    resource = resources.files("unicell.repo.geneformer").joinpath("gene_vocab.json")
    return Path(str(resource)).resolve()


GENEFORMER_BUNDLED_VOCAB = _bundled_geneformer_vocab()
REQUIRED_SCGPT_ARGS = {
    "embsize",
    "nheads",
    "d_hid",
    "nlayers",
    "n_bins",
    "max_seq_len",
    "pad_token",
    "pad_value",
}
CHECKPOINT_ARTIFACTS = {
    "unicell_v1.best.pth",
    "gene_names.pk",
    "celltype_dict.pk",
    "tissue_dict.pk",
    "species_dict.pk",
    "ontoGraph.pk",
    "ontoGraph.graph.gml",
}


class StartupValidationError(ValueError):
    """Raised when startup validation finds one or more blocking problems."""


def _is_primary_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _world_size() -> int:
    try:
        return max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    except ValueError as exc:
        raise StartupValidationError("WORLD_SIZE must be a positive integer.") from exc


def _cells_per_rank(n_obs: int) -> int:
    """Match DistributedSampler's per-rank sample count (drop_last=False)."""
    return int(math.ceil(n_obs / _world_size())) if n_obs else 0


def _validate_batch_count(n_obs: int, batch_size: int, context: str) -> None:
    per_rank = _cells_per_rank(n_obs)
    if per_rank < batch_size:
        raise ValueError(
            f"{context} has too few cells for drop_last=True: total={n_obs}, "
            f"WORLD_SIZE={_world_size()}, per_rank={per_rank}, batch_size={batch_size}."
        )


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _read_json(path: Path, description: str, errors: list[str]) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        errors.append(f"Cannot read {description} JSON '{path}': {exc}")
        return None

    if not isinstance(value, dict):
        errors.append(f"{description} must contain a JSON object: {path}")
        return None
    return value


def _validate_file(
    value: Optional[str],
    option_name: str,
    errors: list[str],
) -> Optional[Path]:
    path = _resolve_path(value)
    if path is None:
        errors.append(f"{option_name} is required.")
        return None
    if not path.exists():
        errors.append(f"{option_name} does not exist: {path}")
    elif not path.is_file():
        errors.append(f"{option_name} must be a file: {path}")
    elif path.stat().st_size == 0:
        errors.append(f"{option_name} is empty: {path}")
    return path


def _validate_directory(
    value: Optional[str],
    option_name: str,
    errors: list[str],
) -> Optional[Path]:
    path = _resolve_path(value)
    if path is None:
        errors.append(f"{option_name} is required.")
        return None
    if not path.exists():
        errors.append(f"{option_name} does not exist: {path}")
    elif not path.is_dir():
        errors.append(f"{option_name} must be a directory: {path}")
    return path


def _validate_vocab(
    vocab_path: Optional[Path],
    description: str,
    errors: list[str],
) -> Optional[Dict[str, int]]:
    if vocab_path is None or not vocab_path.is_file():
        return None

    raw_vocab = _read_json(vocab_path, description, errors)
    if raw_vocab is None:
        return None

    bad_entries = [
        key
        for key, value in raw_vocab.items()
        if not isinstance(key, str)
        or not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ]
    if bad_entries:
        errors.append(
            f"{description} must map string tokens to non-negative integer IDs; "
            f"invalid entries include: {bad_entries[:10]}"
        )
        return None

    token_ids = list(raw_vocab.values())
    if not token_ids:
        errors.append(f"{description} must not be empty: {vocab_path}")
        return None

    if len(set(token_ids)) != len(token_ids):
        errors.append(f"{description} contains duplicate token IDs: {vocab_path}")

    expected_ids = set(range(len(token_ids)))
    if set(token_ids) != expected_ids:
        errors.append(
            f"{description} token IDs must be consecutive from 0 to "
            f"{len(token_ids) - 1}: {vocab_path}"
        )

    return raw_vocab  # type: ignore[return-value]


def _inspect_h5ad(
    path: Path,
    name: str,
    required_obs: Iterable[str],
    errors: list[str],
) -> Optional[Dict[str, Any]]:
    """Read only AnnData metadata and return the fields needed by preflight."""
    try:
        adata = ad.read_h5ad(path, backed="r")
    except Exception as exc:
        errors.append(f"Cannot open {name} h5ad '{path}': {exc}")
        return None

    try:
        missing_obs = sorted(set(required_obs) - set(adata.obs.columns))
        if missing_obs:
            errors.append(f"{name} h5ad is missing obs columns: {missing_obs}")

        if adata.n_obs == 0 or adata.n_vars == 0:
            errors.append(
                f"{name} h5ad must be non-empty; shape is "
                f"({adata.n_obs}, {adata.n_vars})."
            )

        if adata.var_names.has_duplicates:
            duplicates = adata.var_names[adata.var_names.duplicated()].tolist()
            errors.append(
                f"{name} h5ad has duplicate var_names; examples: {duplicates[:10]}"
            )

        for column in required_obs:
            if column in adata.obs.columns and adata.obs[column].isna().any():
                errors.append(f"{name} h5ad obs['{column}'] contains missing values.")

        obs_values = {
            column: set(adata.obs[column].astype(str).tolist())
            for column in required_obs
            if column in adata.obs.columns
        }
        return {
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "var_names": set(map(str, adata.var_names.tolist())),
            "obs_values": obs_values,
        }
    finally:
        # Backed AnnData holds an h5py handle. Close it before model startup.
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _validate_output_directory(args: argparse.Namespace, errors: list[str]) -> Path:
    ckpt_dir = _resolve_path(args.ckpt_dir)
    assert ckpt_dir is not None

    if ckpt_dir.exists() and not ckpt_dir.is_dir():
        errors.append(f"--ckpt_dir must be a directory: {ckpt_dir}")
        return ckpt_dir

    # Every torchrun rank performs preflight before process-group setup, so all
    # ranks must make the same overwrite decision; otherwise one rank can enter
    # init_process_group while rank 0 exits with a validation error.
    if ckpt_dir.exists():
        conflicting = sorted(
            path.name for path in ckpt_dir.iterdir() if path.name in CHECKPOINT_ARTIFACTS
        )
        if conflicting and not args.allow_existing_ckpt_dir:
            errors.append(
                "--ckpt_dir already contains UniCell artifacts "
                f"{conflicting}. Choose a new directory or explicitly pass "
                "--allow_existing_ckpt_dir to permit overwriting generated files."
            )

    writable_parent = _nearest_existing_parent(ckpt_dir)
    if not writable_parent.is_dir() or not os.access(writable_parent, os.W_OK):
        errors.append(f"Checkpoint directory parent is not writable: {writable_parent}")

    return ckpt_dir


def _validate_geneformer(
    args: argparse.Namespace,
    errors: list[str],
    notices: list[str],
) -> Tuple[Optional[Dict[str, int]], Dict[str, Any]]:
    if args.llm_vocab_file is None:
        args.llm_vocab_file = str(GENEFORMER_BUNDLED_VOCAB)
        notices.append(
            "Using the GeneFormer vocabulary bundled with the installed UniCell package."
        )
    model_dir = _validate_directory(args.llm_model_file, "--llm_model_file", errors)
    vocab_path = _validate_file(args.llm_vocab_file, "--llm_vocab_file", errors)
    vocab = _validate_vocab(vocab_path, "GeneFormer vocabulary", errors)
    details: Dict[str, Any] = {}

    if args.llm_args_file is not None:
        notices.append("--llm_args_file is ignored for GeneFormer.")

    if vocab is not None:
        if "<pad>" not in vocab:
            errors.append("GeneFormer vocabulary must contain the '<pad>' token.")

        bundled_errors: list[str] = []
        bundled_vocab = _read_json(
            GENEFORMER_BUNDLED_VOCAB,
            "bundled GeneFormer vocabulary",
            bundled_errors,
        )
        errors.extend(bundled_errors)
        if bundled_vocab is not None and vocab != bundled_vocab:
            errors.append(
                "The current GeneFormer tokenizer in unicell/dataset.py uses the bundled "
                f"vocabulary '{GENEFORMER_BUNDLED_VOCAB}'. --llm_vocab_file must have "
                "the same token-to-ID mapping when the core code is left unchanged."
            )

    if model_dir is not None and model_dir.is_dir():
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            errors.append(
                "GeneFormer --llm_model_file must be a local Hugging Face model "
                f"directory containing config.json: {model_dir}"
            )
        else:
            config = _read_json(config_path, "GeneFormer model config", errors)
            if config is not None:
                details["hidden_size"] = config.get("hidden_size")
                details["model_vocab_size"] = config.get("vocab_size")
                if not isinstance(config.get("hidden_size"), int) or config["hidden_size"] <= 0:
                    errors.append("GeneFormer config.json needs a positive integer 'hidden_size'.")
                if vocab is not None and isinstance(config.get("vocab_size"), int):
                    required_vocab_size = max(vocab.values()) + 1
                    if config["vocab_size"] < required_vocab_size:
                        errors.append(
                            "GeneFormer model vocab_size is smaller than the largest "
                            f"token ID: model={config['vocab_size']}, required={required_vocab_size}."
                        )

        weight_files = []
        for pattern in (
            "pytorch_model*.bin",
            "model*.safetensors",
            "pytorch_model*.bin.index.json",
            "model*.safetensors.index.json",
        ):
            weight_files.extend(model_dir.glob(pattern))
        if not weight_files:
            errors.append(
                "GeneFormer model directory contains no supported local PyTorch or "
                f"safetensors weights: {model_dir}"
            )

    return vocab, details


def _validate_scgpt(
    args: argparse.Namespace,
    errors: list[str],
    notices: list[str],
) -> Tuple[Optional[Dict[str, int]], Dict[str, Any]]:
    model_path = _validate_file(args.llm_model_file, "--llm_model_file", errors)
    vocab_path = _validate_file(args.llm_vocab_file, "--llm_vocab_file", errors)
    args_path = _validate_file(args.llm_args_file, "--llm_args_file", errors)
    vocab = _validate_vocab(vocab_path, "scGPT vocabulary", errors)
    details: Dict[str, Any] = {}

    model_args = None
    if args_path is not None and args_path.is_file():
        model_args = _read_json(args_path, "scGPT args", errors)

    if model_args is not None:
        missing_keys = sorted(REQUIRED_SCGPT_ARGS - set(model_args))
        if missing_keys:
            errors.append(f"scGPT args JSON is missing required keys: {missing_keys}")

        positive_integer_keys = (
            "embsize",
            "nheads",
            "d_hid",
            "nlayers",
            "n_bins",
            "max_seq_len",
        )
        for key in positive_integer_keys:
            if key in model_args and (
                not isinstance(model_args[key], int)
                or isinstance(model_args[key], bool)
                or model_args[key] <= 0
            ):
                errors.append(f"scGPT args['{key}'] must be a positive integer.")

        embsize = model_args.get("embsize")
        nheads = model_args.get("nheads")
        if isinstance(embsize, int) and isinstance(nheads, int) and nheads > 0:
            if embsize % nheads != 0:
                errors.append(
                    f"scGPT embsize ({embsize}) must be divisible by nheads ({nheads})."
                )

        pad_token = model_args.get("pad_token")
        if not isinstance(pad_token, str) or not pad_token:
            errors.append("scGPT args['pad_token'] must be a non-empty string.")
        elif vocab is not None and pad_token not in vocab:
            errors.append(f"scGPT pad token '{pad_token}' is absent from the vocabulary.")

        pad_value = model_args.get("pad_value")
        if not isinstance(pad_value, (int, float)) or isinstance(pad_value, bool):
            errors.append("scGPT args['pad_value'] must be numeric.")

        if vocab is not None and "<cls>" not in vocab:
            errors.append(
                "scGPT vocabulary must contain '<cls>' because the current tokenizer "
                "calls tokenize_and_pad_batch(..., append_cls=True)."
            )

        details.update(
            {
                "embsize": model_args.get("embsize"),
                "nheads": model_args.get("nheads"),
                "n_bins": model_args.get("n_bins"),
                "max_seq_len": model_args.get("max_seq_len"),
            }
        )

    if (
        model_path is not None
        and model_path.is_file()
        and vocab is not None
        and model_args is not None
        and isinstance(model_args.get("embsize"), int)
        and isinstance(model_args.get("nlayers"), int)
    ):
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
        except Exception as exc:
            errors.append(f"Cannot load scGPT checkpoint '{model_path}': {exc}")
        else:
            state_dict = (
                checkpoint.get("model_state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            if not isinstance(state_dict, dict) or not state_dict:
                errors.append(
                    "scGPT checkpoint must contain a non-empty tensor state dict "
                    f"(optionally under 'model_state_dict'): {model_path}"
                )
            else:
                state_dict = {
                    str(key).removeprefix("module."): value
                    for key, value in state_dict.items()
                    if torch.is_tensor(value)
                }
                embedding = state_dict.get("encoder.embedding.weight")
                expected_embedding_shape = (len(vocab), model_args["embsize"])
                if embedding is None or tuple(embedding.shape) != expected_embedding_shape:
                    actual = None if embedding is None else tuple(embedding.shape)
                    errors.append(
                        "scGPT checkpoint embedding is incompatible with vocab/args: "
                        f"expected={expected_embedding_shape}, actual={actual}."
                    )

                missing_attention_layers = []
                attention_style = None
                for layer_index in range(model_args["nlayers"]):
                    fast_key = (
                        f"transformer_encoder.layers.{layer_index}."
                        "self_attn.Wqkv.weight"
                    )
                    torch_key = (
                        f"transformer_encoder.layers.{layer_index}."
                        "self_attn.in_proj_weight"
                    )
                    if fast_key in state_dict:
                        attention_style = attention_style or "Wqkv"
                    elif torch_key in state_dict:
                        attention_style = attention_style or "in_proj"
                    else:
                        missing_attention_layers.append(layer_index)
                if missing_attention_layers:
                    errors.append(
                        "scGPT checkpoint is missing self-attention projection weights "
                        f"for layers: {missing_attention_layers}."
                    )
                details["checkpoint_tensor_keys"] = len(state_dict)
                details["attention_style"] = attention_style

    if model_path is not None and model_path.is_file() and model_path.suffix not in {
        ".pt",
        ".pth",
        ".bin",
    }:
        notices.append(
            f"scGPT checkpoint has an unusual extension '{model_path.suffix}'; "
            "the file will still be passed to torch.load."
        )

    return vocab, details


def _validate_hyperparameters(args: argparse.Namespace, errors: list[str]) -> None:
    positive_values = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "global_layer": args.global_layer,
        "local_layer": args.local_layer,
    }
    if args.input_type == "expr":
        positive_values["output_dim"] = args.output_dim
    for name, value in positive_values.items():
        if value <= 0:
            errors.append(f"--{name} must be greater than zero; got {value}.")

    if args.beta < 0:
        errors.append(f"--beta must be non-negative; got {args.beta}.")
    if not 0 <= args.hidden_layer_dropout < 1:
        errors.append(
            "--hidden_layer_dropout must be in the interval [0, 1); "
            f"got {args.hidden_layer_dropout}."
        )
    if args.seed < 0:
        errors.append(f"--seed must be non-negative; got {args.seed}.")

    annotation_keys = (args.cell_type_key, args.tissue_key, args.species_key)
    if any(not key for key in annotation_keys):
        errors.append("Annotation keys must be non-empty strings.")
    if len(set(annotation_keys)) != len(annotation_keys):
        errors.append(
            "--cell_type_key, --tissue_key, and --species_key must name distinct columns."
        )

    valid_device = (
        args.device in {"auto", "cpu", "cuda"}
        or (
            args.device.startswith("cuda:")
            and args.device.removeprefix("cuda:").isdigit()
        )
    )
    if not valid_device:
        errors.append(
            "--device must be one of auto, cpu, cuda, or cuda:<non-negative index>; "
            f"got {args.device!r}."
        )
    elif args.device.startswith("cuda:") and torch.cuda.is_available():
        device_index = int(args.device.removeprefix("cuda:"))
        if device_index >= torch.cuda.device_count():
            errors.append(
                f"--device={args.device} is unavailable; detected "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )


def validate_startup(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate files, model assets, data schema, and basic compatibility.

    This function is always called by ``main`` before distributed setup and
    before constructing ``scDataset`` or ``HMCN``.
    """
    errors: list[str] = []
    notices: list[str] = []
    required_obs = (args.cell_type_key, args.tissue_key, args.species_key)

    _validate_hyperparameters(args, errors)

    train_path = _validate_file(args.train_h5ad, "--train_h5ad", errors)
    eval_path = _validate_file(args.eval_h5ad, "--eval_h5ad", errors)
    ckpt_dir = _validate_output_directory(args, errors)

    train_info = (
        _inspect_h5ad(train_path, "Training", required_obs, errors)
        if train_path is not None and train_path.is_file()
        else None
    )
    eval_info = (
        _inspect_h5ad(eval_path, "Evaluation", required_obs, errors)
        if eval_path is not None and eval_path.is_file()
        else None
    )

    if train_path is not None and eval_path is not None and train_path == eval_path:
        notices.append("Training and evaluation paths refer to the same h5ad file.")

    if train_info is not None:
        per_rank_cells = _cells_per_rank(train_info["n_obs"])
        if per_rank_cells < args.batch_size:
            errors.append(
                "Training h5ad has too few cells per process while the existing "
                "Trainer uses drop_last=True: "
                f"n_obs={train_info['n_obs']}, WORLD_SIZE={_world_size()}, "
                f"per_rank={per_rank_cells}, batch_size={args.batch_size}."
            )

    if train_info is not None and eval_info is not None:
        missing_eval_genes = train_info["var_names"] - eval_info["var_names"]
        if missing_eval_genes:
            examples = sorted(missing_eval_genes)[:10]
            errors.append(
                "Evaluation data does not contain every training gene: "
                f"missing={len(missing_eval_genes)}, examples={examples}."
            )

        for key, display_name in (
            (args.tissue_key, "tissue"),
            (args.species_key, "species"),
        ):
            train_values = train_info["obs_values"].get(key, set())
            eval_values = eval_info["obs_values"].get(key, set())
            unseen = sorted(eval_values - train_values)
            if unseen:
                errors.append(
                    f"Evaluation data contains {display_name} labels absent from "
                    f"training data: {unseen[:20]}"
                )

    vocab: Optional[Dict[str, int]] = None
    backbone_details: Dict[str, Any] = {}
    if args.input_type == "GeneFormer":
        vocab, backbone_details = _validate_geneformer(args, errors, notices)
    elif args.input_type == "scGPT":
        vocab, backbone_details = _validate_scgpt(args, errors, notices)
    elif args.input_type == "expr":
        supplied = [
            name
            for name, value in (
                ("--llm_model_file", args.llm_model_file),
                ("--llm_vocab_file", args.llm_vocab_file),
                ("--llm_args_file", args.llm_args_file),
            )
            if value is not None
        ]
        if supplied:
            notices.append(f"The expr input type ignores: {', '.join(supplied)}.")

    gene_overlap: Optional[Dict[str, Any]] = None
    if vocab is not None and train_info is not None:
        special_tokens = {token for token in vocab if token.startswith("<") and token.endswith(">")}
        gene_tokens = set(vocab) - special_tokens
        overlap = train_info["var_names"] & gene_tokens
        overlap_fraction = len(overlap) / max(len(train_info["var_names"]), 1)
        gene_overlap = {
            "count": len(overlap),
            "input_gene_count": train_info["n_vars"],
            "fraction": overlap_fraction,
        }
        if not overlap:
            errors.append(
                f"No training var_names overlap the {args.input_type} vocabulary. "
                "Check whether var_names use the gene identifiers expected by the checkpoint."
            )
        elif len(overlap) < 100 or overlap_fraction < 0.05:
            notices.append(
                f"Only {len(overlap)}/{train_info['n_vars']} training genes "
                f"({overlap_fraction:.2%}) overlap the {args.input_type} vocabulary."
            )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        errors.append(f"--device={args.device} was requested but CUDA is unavailable.")

    if errors:
        formatted = "\n".join(f"  - {message}" for message in errors)
        raise StartupValidationError(f"Startup validation failed:\n{formatted}")

    # Normalize paths after successful validation. These exact strings are saved
    # into checkpoint metadata by the existing Trainer and reused by prediction.
    args.train_h5ad = str(train_path)
    args.eval_h5ad = str(eval_path)
    args.ckpt_dir = str(ckpt_dir)
    if args.input_type != "expr":
        args.llm_model_file = str(_resolve_path(args.llm_model_file))
        args.llm_vocab_file = str(_resolve_path(args.llm_vocab_file))
        args.llm_args_file = (
            str(_resolve_path(args.llm_args_file))
            if args.llm_args_file is not None and args.input_type == "scGPT"
            else None
        )
    else:
        args.llm_model_file = None
        args.llm_vocab_file = None
        args.llm_args_file = None

    report = {
        "input_type": args.input_type,
        "world_size": _world_size(),
        "train_shape": [train_info["n_obs"], train_info["n_vars"]]
        if train_info is not None
        else None,
        "eval_shape": [eval_info["n_obs"], eval_info["n_vars"]]
        if eval_info is not None
        else None,
        "gene_overlap": gene_overlap,
        "backbone": backbone_details,
        "notices": notices,
    }

    if _is_primary_process():
        print("\n=== Startup validation ===")
        print(f"Input type : {args.input_type}")
        print(f"Train data : {args.train_h5ad}  shape={report['train_shape']}")
        print(f"Eval data  : {args.eval_h5ad}  shape={report['eval_shape']}")
        print(f"Ckpt dir   : {args.ckpt_dir}")
        if gene_overlap is not None:
            print(
                "Gene/vocab : "
                f"{gene_overlap['count']}/{gene_overlap['input_gene_count']} "
                f"({gene_overlap['fraction']:.2%})"
            )
        for message in notices:
            print(f"[WARN] {message}")
        print("Status     : PASS")
        print("==========================\n")

    return report


def setup_distributed(requested_device: str) -> Tuple[bool, int, str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if requested_device == "cpu" or not torch.cuda.is_available():
            raise RuntimeError(
                "The existing UnicellTrainer DDP path requires CUDA. "
                "Use one process for CPU training."
            )
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        if local_rank == 0:
            print(f"[Init DDP] WORLD_SIZE={world_size}, backend=nccl")
        return True, local_rank, device

    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda":
        device = "cuda:0"
    else:
        device = requested_device

    return False, 0, device


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_run_config(
    args: argparse.Namespace,
    validation_report: Dict[str, Any],
    device: str,
    ddp_train: bool,
) -> None:
    payload = {
        "arguments": vars(args),
        "resolved_device": device,
        "ddp_train": ddp_train,
        "validation": validation_report,
    }
    config_path = Path(args.ckpt_dir) / "run_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _copy_backbone_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def _prepare_portable_backbone_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    """Copy lightweight backbone configuration beside the UniCell checkpoint.

    The trained UniCell state dict already contains all backbone parameters.  A
    portable inference bundle therefore needs only the files required to
    reconstruct the architecture before loading that state dict.
    """
    if args.input_type == "expr":
        return {}

    bundle_dir = Path(args.ckpt_dir) / "backbone"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "input_type": args.input_type,
        "contains_pretrained_weights": False,
        "note": (
            "Backbone parameters are stored inside unicell_v1.best.pth; this "
            "directory contains only architecture/tokenizer configuration."
        ),
    }

    if args.input_type == "GeneFormer":
        source_model_dir = Path(args.llm_model_file)
        _copy_backbone_asset(source_model_dir / "config.json", bundle_dir / "config.json")
        _copy_backbone_asset(Path(args.llm_vocab_file), bundle_dir / "gene_vocab.json")
        metadata = {
            "llm_model_file": "backbone",
            "llm_vocab_file": "backbone/gene_vocab.json",
            "llm_args_file": None,
        }
        manifest["files"] = ["config.json", "gene_vocab.json"]
        manifest["source_model_name"] = source_model_dir.name
    else:
        _copy_backbone_asset(Path(args.llm_vocab_file), bundle_dir / "vocab.json")
        _copy_backbone_asset(Path(args.llm_args_file), bundle_dir / "args.json")
        metadata = {
            "llm_model_file": None,
            "llm_vocab_file": "backbone/vocab.json",
            "llm_args_file": "backbone/args.json",
        }
        manifest["files"] = ["vocab.json", "args.json"]
        manifest["source_model_name"] = Path(args.llm_model_file).name

    metadata.update(
        {
            "backbone_bundle_version": 1,
            "backbone_init_from_config": True,
        }
    )

    manifest_path = bundle_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f">>> Portable backbone config bundle: {bundle_dir}")
    return metadata


def _build_dataset(
    path: str,
    args: argparse.Namespace,
    trained: bool,
) -> "scDataset":
    # Delay core imports so --help and --validate_only can run without loading
    # every vendored foundation-model dependency.
    from unicell.scDataset import scDataset

    return scDataset(
        data_path=path,
        cell_type_key=args.cell_type_key,
        tissue_key=args.tissue_key,
        species_key=args.species_key,
        trained=trained,
        highly_variable_genes=trained,
        llm_vocab=args.llm_vocab_file,
        llm_args=args.llm_args_file,
    )


def _align_evaluation_dataset(
    sc_train: "scDataset",
    sc_eval: "scDataset",
    args: argparse.Namespace,
) -> None:
    missing_eval_genes = sc_train.adata.var_names.difference(sc_eval.adata.var_names)
    if len(missing_eval_genes):
        raise ValueError(
            f"Evaluation data is missing {len(missing_eval_genes)} training genes; "
            f"examples={missing_eval_genes[:10].tolist()}."
        )

    sc_eval.adata = sc_eval.adata[:, sc_train.adata.var_names].copy()
    sc_eval.ontograph = sc_train.ontograph
    sc_eval.cell_type_index = sc_eval.get_cell_type_index()

    unknown_cell_types = sorted(
        {
            str(label)
            for label, index in zip(
                sc_eval.adata.obs[args.cell_type_key].tolist(),
                sc_eval.cell_type_index,
            )
            if index < 0
        }
    )
    if unknown_cell_types and _is_primary_process():
        warnings.warn(
            "Evaluation contains cell-type ontology IDs outside the ontology graph "
            f"constructed from training labels: {unknown_cell_types[:20]}",
            RuntimeWarning,
        )

    eval_tissues = sc_eval.adata.obs[args.tissue_key].astype(str)
    eval_species = sc_eval.adata.obs[args.species_key].astype(str)
    unseen_tissues = sorted(set(eval_tissues) - set(sc_train.tissue_label_dict))
    unseen_species = sorted(set(eval_species) - set(sc_train.species_label_dict))
    if unseen_tissues or unseen_species:
        raise ValueError(
            "Evaluation data contains labels not seen after training-data ontology "
            f"filtering: tissue={unseen_tissues}, species={unseen_species}"
        )

    sc_eval.tissue_label_dict = sc_train.tissue_label_dict
    sc_eval.species_label_dict = sc_train.species_label_dict
    sc_eval.tissue_index = np.asarray(
        [sc_train.tissue_label_dict[value] for value in eval_tissues],
        dtype=np.int64,
    )
    sc_eval.species_index = np.asarray(
        [sc_train.species_label_dict[value] for value in eval_species],
        dtype=np.int64,
    )


def train_with_backbone(
    args: argparse.Namespace,
    validation_report: Dict[str, Any],
    ddp_train: bool,
    local_rank: int,
    device: str,
) -> None:
    from unicell.trainer import UnicellTrainer

    is_main_process = not ddp_train or dist.get_rank() == 0

    if is_main_process:
        Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
        _save_run_config(args, validation_report, device, ddp_train)
    if ddp_train:
        dist.barrier()

    if is_main_process:
        print(f">>> Loading training data: {args.train_h5ad}")
    sc_train = _build_dataset(args.train_h5ad, args, trained=True)
    sc_train.adata = sc_train.adata.copy()

    if sc_train.adata.n_obs == 0:
        raise ValueError("No training cells remain after UniCell ontology/QC filtering.")
    _validate_batch_count(
        sc_train.adata.n_obs,
        args.batch_size,
        "Training data after UniCell ontology/QC filtering",
    )

    if is_main_process:
        print(f">>> Saving gene names and ontology graph: {args.ckpt_dir}")
        with (Path(args.ckpt_dir) / "gene_names.pk").open("wb") as handle:
            pickle.dump(sc_train.adata.var_names, handle)
        sc_train.ontograph.pickle(args.ckpt_dir)
    if ddp_train:
        dist.barrier()

    if is_main_process:
        print(f">>> Loading evaluation data: {args.eval_h5ad}")
    sc_eval = _build_dataset(args.eval_h5ad, args, trained=False)
    _align_evaluation_dataset(sc_train, sc_eval, args)

    input_dim = sc_train.adata.n_vars if args.input_type == "expr" else None
    output_dim = args.output_dim if args.input_type == "expr" else None

    portable_metadata = None
    if is_main_process:
        portable_metadata = _prepare_portable_backbone_bundle(args)
    if ddp_train:
        payload = [portable_metadata]
        dist.broadcast_object_list(payload, src=0)
        portable_metadata = payload[0]

    if is_main_process:
        print(f">>> Initializing {args.input_type} + UniCell trainer")
    trainer = UnicellTrainer(
        sc_train,
        input_type=args.input_type,
        input_dim=input_dim,
        output_dim=output_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        beta=args.beta,
        device=device,
        global_layer=args.global_layer,
        local_layer=args.local_layer,
        hidden_layer_dropout=args.hidden_layer_dropout,
        ckpt_dir=args.ckpt_dir,
        ddp_train=ddp_train,
        save_epoch=args.save_epoch,
        local_rank=local_rank,
        llm_model_file=args.llm_model_file,
        llm_vocab_file=args.llm_vocab_file,
        llm_args_file=args.llm_args_file,
        checkpoint_metadata_overrides=portable_metadata,
    )

    if is_main_process:
        print(">>> Starting training")
    trainer.train(scdata_test=sc_eval)

    if is_main_process:
        print(f">>> Training finished. Checkpoint directory: {args.ckpt_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the existing UniCell implementation with expr, GeneFormer, "
            "or scGPT input while automatically validating assets and data."
        )
    )
    parser.add_argument(
        "--input_type",
        choices=SUPPORTED_INPUT_TYPES,
        required=True,
        help="Existing UniCell encoder/input branch to activate (case-sensitive).",
    )
    parser.add_argument("--train_h5ad", required=True, help="Training AnnData file.")
    parser.add_argument("--eval_h5ad", required=True, help="Evaluation AnnData file.")
    parser.add_argument("--ckpt_dir", required=True, help="Output checkpoint directory.")

    parser.add_argument(
        "--llm_model_file",
        default=None,
        help=(
            "GeneFormer Hugging Face model directory, or scGPT checkpoint file. "
            "Not used for expr."
        ),
    )
    parser.add_argument(
        "--llm_vocab_file",
        default=None,
        help=(
            "scGPT token-to-ID JSON vocabulary, or an optional GeneFormer "
            "override that must exactly match UniCell's bundled vocabulary."
        ),
    )
    parser.add_argument(
        "--llm_args_file",
        default=None,
        help="scGPT args.json. Not used for expr or GeneFormer.",
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--global_layer", type=int, default=128)
    parser.add_argument("--local_layer", type=int, default=64)
    parser.add_argument("--hidden_layer_dropout", type=float, default=0.1)
    parser.add_argument(
        "--output_dim",
        type=int,
        default=512,
        help="Encoder projection size for input_type=expr; ignored by foundation encoders.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a CUDA device such as cuda:1.",
    )

    parser.add_argument("--cell_type_key", default="cell_type_ontology_term_id")
    parser.add_argument("--tissue_key", default="general_tissue")
    parser.add_argument("--species_key", default="organism")
    parser.add_argument("--save_epoch", action="store_true")
    parser.add_argument(
        "--allow_existing_ckpt_dir",
        action="store_true",
        help="Allow generated checkpoint artifacts in an existing directory to be overwritten.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Run the automatic startup checks and exit before model/data initialization.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validation_report = validate_startup(args)
    except StartupValidationError as exc:
        parser.error(str(exc))

    if args.validate_only:
        if _is_primary_process():
            print("Validation-only mode completed; training was not started.")
        return

    ddp_train = False
    try:
        ddp_train, local_rank, device = setup_distributed(args.device)
        set_random_seed(args.seed)
        train_with_backbone(
            args=args,
            validation_report=validation_report,
            ddp_train=ddp_train,
            local_rank=local_rank,
            device=device,
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
