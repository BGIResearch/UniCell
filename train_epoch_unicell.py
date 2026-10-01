#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 1: Train UniCell with fixed eval_h5ad, but DIFFERENT train h5ad PER EPOCH.

Goals:
- Use a fixed eval_h5ad (--eval_h5ad)
- At epoch i, use:
  /path/to/sampled_h5ads/seed_i/train.h5ad

The trainer / loss / model architecture otherwise remain unchanged (no changes to unicell/* source code).
"""

import os
import argparse
import pickle
import time
import copy
import json

import torch
import torch.distributed as dist
import scanpy as sc

from unicell.scDataset import scDataset
from unicell.trainer import UnicellTrainer
from unicell.dataset import HMCNDataset
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler


# ======================
# Default configuration (adjust as needed)
# ======================
PREFIX = "last_version_gml"
CKPT_DIR = os.path.join("models", PREFIX)

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
BETA = 0.1
GLOBAL_LAYER = 128
LOCAL_LAYER = 64
HIDDEN_LAYER_DROPOUT = 0.1

CELL_TYPE_KEY = "cell_type_ontology_term_id"
TISSUE_KEY = "general_tissue"
SPECIES_KEY = "organism"


def setup_distributed():
    """
    Determine from environment variables whether to initialize distributed training.
    Returns:
        ddp_train: bool
        local_rank: int
        device: str
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        print(f"[Init DDP] RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"

        ddp_train = world_size > 1
    else:
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ddp_train = False

    return ddp_train, local_rank, device


def _seed_train_path(seed_root: str, seed_i: int) -> str:
    return os.path.join(seed_root, f"seed_{seed_i}", "train.h5ad")


def _standardize_sc_train_to_ref(sc_train_epoch: scDataset, ref_train: scDataset):
    """
    Align each epoch's scDataset with the reference (seed_0) in terms of:
    - Gene set and order (using ref_train.adata.var_names)
    - ontograph (using ref_train.ontograph)
    - cell_type_index (re-encoded using the reference ontograph)
    - tissue/species indices (re-encoded using the reference dictionaries to ensure consistent class ordering)
    """
    # 1) genes align (order fixed)
    ref_genes = ref_train.adata.var_names
    sc_train_epoch.adata = sc_train_epoch.adata[:, ref_genes]

    # 2) use ref ontograph + apply same filtering rules
    sc_train_epoch.ontograph = ref_train.ontograph

    # obsolete replace + name mapping
    sc_train_epoch.adata.obs[CELL_TYPE_KEY] = [
        sc_train_epoch.ontograph.obsolete_dict.get(c, c)
        for c in sc_train_epoch.adata.obs[CELL_TYPE_KEY]
    ]
    sc_train_epoch.adata.obs["cell_type"] = [
        sc_train_epoch.ontograph.id2name.get(c, "unknown")
        for c in sc_train_epoch.adata.obs[CELL_TYPE_KEY]
    ]

    # same filtering as scDataset(trained=True) did
    sc_train_epoch.adata = sc_train_epoch.adata[
        ~sc_train_epoch.adata.obs[CELL_TYPE_KEY].isin([sc_train_epoch.ontograph.common_ancestor]), :
    ]
    sc_train_epoch.adata = sc_train_epoch.adata[
        sc_train_epoch.adata.obs[CELL_TYPE_KEY].isin(sc_train_epoch.ontograph.vocab.keys()), :
    ]

    # 3) re-build cell_type_index (needs ontograph)
    sc_train_epoch.cell_type_key = CELL_TYPE_KEY
    sc_train_epoch.cell_type_index = sc_train_epoch.get_cell_type_index()

    # 4) tissue/species recode to ref dict ordering (important!)
    # tissue
    sc_train_epoch.tissue_key = TISSUE_KEY
    sc_train_epoch.adata.obs[TISSUE_KEY] = sc_train_epoch.adata.obs[TISSUE_KEY].astype("category")
    sc_train_epoch.tissue_label_dict = ref_train.tissue_label_dict
    tissue_codes = sc_train_epoch.adata.obs[TISSUE_KEY].astype(str).map(ref_train.tissue_label_dict)
    if tissue_codes.isnull().any():
        missing = sorted(set(sc_train_epoch.adata.obs[TISSUE_KEY].astype(str)[tissue_codes.isnull()].tolist()))
        raise ValueError(f"[Epoch Train] tissue labels not in reference dict: {missing[:20]} (show first 20)")
    sc_train_epoch.tissue_index = tissue_codes.astype(int).to_numpy()

    # species
    sc_train_epoch.species_key = SPECIES_KEY
    sc_train_epoch.adata.obs[SPECIES_KEY] = sc_train_epoch.adata.obs[SPECIES_KEY].astype("category")
    sc_train_epoch.species_label_dict = ref_train.species_label_dict
    species_codes = sc_train_epoch.adata.obs[SPECIES_KEY].astype(str).map(ref_train.species_label_dict)
    if species_codes.isnull().any():
        missing = sorted(set(sc_train_epoch.adata.obs[SPECIES_KEY].astype(str)[species_codes.isnull()].tolist()))
        raise ValueError(f"[Epoch Train] species labels not in reference dict: {missing[:20]} (show first 20)")
    sc_train_epoch.species_index = species_codes.astype(int).to_numpy()

    return sc_train_epoch


def _reset_trainer_dataloader(trainer: UnicellTrainer, sc_train_epoch: scDataset):
    """
    Replace only the trainer's dataloader (model/optimizer/loss remain unchanged)
    """
    dataset = HMCNDataset(sc_train_epoch, trainer.input_type)
    sampler = RandomSampler(dataset) if not trainer.ddp_train else DistributedSampler(dataset)

    trainer.dataset = dataset
    trainer.dataloader = DataLoader(
        dataset,
        batch_size=trainer.batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=trainer.collate_fn
    )


def train_unicell(
    ddp_train: bool,
    local_rank: int,
    device: str,
    eval_h5ad: str,
    seed_root: str,
    seed_start: int,
):
    # Create the ckpt directory only on rank0 to avoid concurrent writes, then synchronize with a barrier
    if local_rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
    if ddp_train:
        dist.barrier()

    # ========== 0. reference train (seed_start) ==========
    ref_train_h5ad = _seed_train_path(seed_root, seed_start)
    if local_rank == 0:
        print(">>> Loading reference training data (for fixed genes/ontograph/dicts):", ref_train_h5ad)

    ref_train = scDataset(
        data_path=ref_train_h5ad,
        cell_type_key=CELL_TYPE_KEY,
        tissue_key=TISSUE_KEY,
        species_key=SPECIES_KEY,
        trained=True,
        highly_variable_genes=True,   # reference uses HVG (keep your original behavior)
    )
    input_dim = ref_train.adata.shape[1]

    # ========== 1. Save gene names and the ontology graph (rank0 only) ==========
    if local_rank == 0:
        print(">>> Saving gene names and ontology graph to:", CKPT_DIR)
        with open(os.path.join(CKPT_DIR, "gene_names.pk"), "wb") as w1:
            pickle.dump(ref_train.adata.var_names, w1)
        ref_train.ontograph.pickle(CKPT_DIR)

    if ddp_train:
        dist.barrier()

    # ========== 2. Load validation data (fixed across epochs) ==========
    if local_rank == 0:
        print(">>> Loading eval data (fixed):", eval_h5ad)

    sc_eval = scDataset(
        adata=None,
        data_path=eval_h5ad,
        cell_type_key=CELL_TYPE_KEY,
        tissue_key=TISSUE_KEY,
        species_key=SPECIES_KEY,
        trained=False,
        highly_variable_genes=False,
    )

    # Align eval with the reference gene order + ontograph
    if local_rank == 0:
        print(">>> Aligning eval genes to reference training genes ...")
    sc_eval.adata = sc_eval.adata[:, ref_train.adata.var_names]
    sc_eval.ontograph = ref_train.ontograph
    sc_eval.cell_type_index = sc_eval.get_cell_type_index()

    # ========== 3. Initialize Trainer (once) ==========
    if local_rank == 0:
        print(">>> Initializing UniCell trainer (once) ...")

    trainer = UnicellTrainer(
        ref_train,
        input_type="expr",
        input_dim=input_dim,
        output_dim=512,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,   # Note: we call trainer.train_one_epoch from an external loop for each epoch instead of calling trainer.train directly
        beta=BETA,
        device=device,
        global_layer=GLOBAL_LAYER,
        local_layer=LOCAL_LAYER,
        hidden_layer_dropout=HIDDEN_LAYER_DROPOUT,
        ckpt_dir=CKPT_DIR,
        ddp_train=ddp_train,
        local_rank=local_rank,
    )

    # ========== 4. External epoch loop: switch to a different train.h5ad for each epoch ==========
    if local_rank == 0:
        print(">>> Start training with per-epoch train.h5ad switching ...")

    if local_rank == 0:
        total_start = time.time()
        epoch_times = []

    for epoch in range(NUM_EPOCHS):
        seed_i = seed_start + epoch
        train_h5ad_epoch = _seed_train_path(seed_root, seed_i)

        if local_rank == 0:
            print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}] Using train h5ad: {train_h5ad_epoch}")

        # load epoch train (avoid HVG re-selection; then subset to ref HVG)
        sc_train_epoch = scDataset(
            data_path=train_h5ad_epoch,
            cell_type_key=CELL_TYPE_KEY,
            tissue_key=TISSUE_KEY,
            species_key=SPECIES_KEY,
            trained=True,
            highly_variable_genes=False,   # important: do NOT re-select HVG here
        )
        sc_train_epoch = _standardize_sc_train_to_ref(sc_train_epoch, ref_train)

        # swap trainer dataloader
        _reset_trainer_dataloader(trainer, sc_train_epoch)

        # train one epoch
        trainer.model.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.time()
        
        epoch_loss = trainer.train_one_epoch(epoch)

        # eval (fixed)
        cls_eval_res = trainer.predict(sc_eval, batch_size=trainer.batch_size)
        acc = float(cls_eval_res["accuracy"])
        f1 = float(cls_eval_res["macro_f1"])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        if local_rank == 0:
            epoch_times.append(epoch_time)
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}]/{epoch_time:.2f}s, "
                f"Train Loss: {epoch_loss:.6f}, eval_Acc: {acc:.4f}, macro_f1: {f1:.4f}."
            )

        # mimic trainer.train() best-model selection behavior (by f1 primarily)
        if f1 >= trainer.best_f1:
            trainer.best_f1 = f1
            trainer.best_loss = epoch_loss
            trainer.best_model = copy.deepcopy(trainer.model)
            trainer.best_epoch = epoch

    # ========== 5. mimic trainer.train() saving (rank0 only) ==========
    if local_rank == 0:
        total_end = time.time()
        total_time = total_end - total_start
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if len(epoch_times) > 0 else 0.0

        print("\n===== Training Time Summary =====")
        print(f"Total epochs      : {NUM_EPOCHS}")
        print(f"Total train time  : {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
        print(f"Avg time / epoch  : {avg_epoch_time:.2f} seconds")
        print(f"Best epoch (by f1): {trainer.best_epoch + 1}")
        print("=================================\n")

        time_stats = {
            "total_epochs": int(NUM_EPOCHS),
            "total_time_seconds": float(total_time),
            "total_time_minutes": float(total_time / 60.0),
            "avg_epoch_time_seconds": float(avg_epoch_time),
            "epoch_times_seconds": [float(t) for t in epoch_times],
            "best_epoch": int(trainer.best_epoch + 1),
        }
        time_stats_path = os.path.join(CKPT_DIR, "train_time_summary.json")
        with open(time_stats_path, "w", encoding="utf-8") as f:
            json.dump(time_stats, f, indent=2, ensure_ascii=False)
        print(f"[TimeSummary] Saved training time stats to: {time_stats_path}")

        metadata = {
            "input_type": trainer.input_type,
            "input_dim": trainer.input_dim,
            "output_dim": trainer.output_dim,
            "global_layer": trainer.global_layer,
            "local_layer": trainer.local_layer,
            "hidden_layer_dropout": trainer.hidden_layer_dropout,
            "llm_model_file": trainer.llm_model_file,
            "llm_vocab_file": trainer.llm_vocab_file,
            "llm_args_file": trainer.llm_args_file,
            "tissue_key": trainer.tissue_key,
            "species_key": trainer.species_key,
        }

        checkpoint = {
            "model_state_dict": trainer.best_model.state_dict(),
            "metadata": metadata,
        }
        best_path = os.path.join(CKPT_DIR, "unicell_v1.best.pth")
        torch.save(checkpoint, best_path)
        print(">>> Training finished. Best checkpoint saved to:", best_path)


def main():
    parser = argparse.ArgumentParser()

    # Fixed eval input
    parser.add_argument(
        "--eval_h5ad",
        type=str,
        required=True,
        help="Evaluation/validation h5ad file (fixed across all epochs)."
    )

    # Per-epoch training directory template
    parser.add_argument(
        "--seed_root",
        type=str,
        required=True,
        help="Root dir containing seed_i/train.h5ad."
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=0,
        help="Start seed index. Epoch 0 uses seed_start; epoch e uses seed_start+e."
    )

    args = parser.parse_args()

    ddp_train, local_rank, device = setup_distributed()

    train_unicell(
        ddp_train=ddp_train,
        local_rank=local_rank,
        device=device,
        eval_h5ad=args.eval_h5ad,
        seed_root=args.seed_root,
        seed_start=args.seed_start,
    )

    if ddp_train:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
