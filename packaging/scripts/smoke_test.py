#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
from importlib import metadata, resources
import os
import pickle
import platform
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unicell-matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/unicell-numba")


def check_imports() -> None:
    import anndata
    import scanpy
    import torch
    import torchtext
    import unicell

    print(f"architecture={platform.machine()}")
    print(f"unicell={unicell.__version__}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torchtext={torchtext.__version__}")
    print(f"scanpy={scanpy.__version__}")
    print(f"anndata={anndata.__version__}")

    from predict import main as predict_main
    from predict_by_sample_chunks import main as chunk_main
    from train_Unicell import main as train_main
    from unicell.cli.train_backbone import main as backbone_train_main

    assert callable(predict_main)
    assert callable(chunk_main)
    assert callable(train_main)
    assert callable(backbone_train_main)

    discovered_entry_points = metadata.entry_points()
    if hasattr(discovered_entry_points, "select"):
        console_scripts = discovered_entry_points.select(group="console_scripts")
    else:
        console_scripts = discovered_entry_points.get("console_scripts", ())
    entry_points = {entry_point.name for entry_point in console_scripts}
    assert "unicell-train-backbone" in entry_points

    geneformer_vocab = resources.files("unicell.repo.geneformer").joinpath(
        "gene_vocab.json"
    )
    scgpt_vocab = resources.files("unicell.repo.scgpt.tokenizer").joinpath(
        "default_gene_vocab.json"
    )
    assert geneformer_vocab.is_file()
    assert scgpt_vocab.is_file()
    print("entrypoint_imports=OK")
    print("foundation_resources=OK")


def check_checkpoint(checkpoint_dir: Path, instantiate_model: bool = False) -> None:
    import torch

    from unicell.hmcn import HMCN
    from unicell.utils.utils import load_ontograph

    required = (
        "unicell_v1.best.pth",
        "gene_names.pk",
        "ontoGraph.pk",
        "ontoGraph.graph.gml",
        "celltype_dict.pk",
        "tissue_dict.pk",
        "species_dict.pk",
    )
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint files: {missing}")

    checkpoint = torch.load(checkpoint_dir / "unicell_v1.best.pth", map_location="cpu")
    metadata = checkpoint["metadata"]
    if metadata["input_type"] != "expr":
        raise ValueError(f"Expected expr checkpoint, got {metadata['input_type']!r}")

    with (checkpoint_dir / "celltype_dict.pk").open("rb") as handle:
        celltype_dict = pickle.load(handle)
    with (checkpoint_dir / "tissue_dict.pk").open("rb") as handle:
        tissue_dict = pickle.load(handle)
    with (checkpoint_dir / "species_dict.pk").open("rb") as handle:
        species_dict = pickle.load(handle)

    ontograph = load_ontograph(str(checkpoint_dir))
    arrays = ontograph.hierarchical_array
    num_classes = len(arrays)
    hierarchical_class = [array.shape[1] for array in arrays]
    hierarchical_depth = [metadata["global_layer"] if index > 0 else 0 for index in range(num_classes)]
    global2local = [metadata["local_layer"] if index > 0 else 0 for index in range(num_classes)]

    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    required_state_keys = (
        "encoder.linear.0.weight",
        "global_cls.weight",
        "tissue_cls.weight",
        "species_cls.weight",
    )
    missing_state_keys = [key for key in required_state_keys if key not in state_dict]
    if missing_state_keys:
        raise KeyError(f"Missing model state keys: {missing_state_keys}")

    parameter_count = sum(tensor.numel() for tensor in state_dict.values())
    print(f"checkpoint_input_type={metadata['input_type']}")
    print(f"checkpoint_parameters={parameter_count}")
    print("checkpoint_structure=OK")

    if not instantiate_model:
        del checkpoint, state_dict
        gc.collect()
        return

    model = HMCN(
        input_type=metadata["input_type"],
        input_dim=metadata["input_dim"],
        output_dim=metadata["output_dim"],
        num_classes=num_classes,
        hierarchical_depth=hierarchical_depth,
        global2local=global2local,
        hierarchical_class=hierarchical_class,
        hidden_layer_dropout=metadata["hidden_layer_dropout"],
        cls_num=len(celltype_dict),
        tissue_cls_num=len(tissue_dict),
        species_cls_num=len(species_dict),
        llm_model_file=metadata["llm_model_file"],
        llm_vocab_file=metadata["llm_vocab_file"],
        llm_args_file=metadata["llm_args_file"],
    )
    model.load_state_dict(state_dict, strict=True)
    print("checkpoint_load=OK")
    del model, checkpoint
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a UniCell wheel installation.")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--instantiate-model",
        action="store_true",
        help="Also instantiate HMCN and copy all checkpoint tensors; needs about 1 GB extra RAM.",
    )
    args = parser.parse_args()

    check_imports()
    if args.checkpoint_dir is not None:
        check_checkpoint(args.checkpoint_dir.resolve(), instantiate_model=args.instantiate_model)


if __name__ == "__main__":
    main()
