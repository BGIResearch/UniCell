#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 1: Train UniCell with manually specified train/eval h5ad files.

Single GPU:
    python train_Unicell.py --train_h5ad /path/to/train.h5ad --eval_h5ad /path/to/eval.h5ad

Multiple GPUs (DDP):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_Unicell.py \
        --train_h5ad /path/to/train.h5ad \
        --eval_h5ad /path/to/eval.h5ad

Notes:
- Only the data input method changes: automatic splitting of one h5ad -> manually specified train/eval h5ad files
- Training logic, trainer, loss, and model architecture remain unchanged
- In DDP, only rank0 writes gene_names / ontograph; other ranks read them after the barrier
"""

import os
import argparse
import pickle

import torch
import torch.distributed as dist
import scanpy as sc
import numpy as np

from unicell.scDataset import scDataset
from unicell.trainer import UnicellTrainer


# ======================
# Default configuration (adjust as needed)
# ======================
PREFIX = "checkpoints"
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


def train_unicell(
    ddp_train: bool,
    local_rank: int,
    device: str,
    train_h5ad: str,
    eval_h5ad: str,
):
    # Create the ckpt directory only on rank0 to avoid concurrent writes, then synchronize with a barrier
    if local_rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
    if ddp_train:
        dist.barrier()

    # ========== 1. Load training data ==========
    if local_rank == 0:
        print(">>> Loading training data:", train_h5ad)

    sc_train = scDataset(
        data_path=train_h5ad,
        cell_type_key=CELL_TYPE_KEY,
        tissue_key=TISSUE_KEY,
        species_key=SPECIES_KEY,
        trained=True,
        highly_variable_genes=True,
    )

    input_dim = sc_train.adata.shape[1]

    # ========== 2. Save gene names and the ontology graph (rank0 only) ==========
    if local_rank == 0:
        print(">>> Saving gene names and ontology graph to:", CKPT_DIR)
        with open(os.path.join(CKPT_DIR, "gene_names.pk"), "wb") as w1:
            pickle.dump(sc_train.adata.var_names, w1)
        sc_train.ontograph.pickle(CKPT_DIR)

    if ddp_train:
        dist.barrier()

    # ========== 3. Load validation data ==========
    if local_rank == 0:
        print(">>> Loading eval data:", eval_h5ad)

    sc_eval = scDataset(
        adata=None,
        data_path=eval_h5ad,
        cell_type_key=CELL_TYPE_KEY,
        tissue_key=TISSUE_KEY,
        species_key=SPECIES_KEY,
        trained=False,
        highly_variable_genes=False,
    )

    # Check that validation data contains all genes required for training and align their order
    if local_rank == 0:
        print(">>> Aligning eval genes to training genes ...")
    missing_eval_genes = sc_train.adata.var_names.difference(sc_eval.adata.var_names)
    if len(missing_eval_genes):
        raise ValueError(
            f"Evaluation data is missing {len(missing_eval_genes)} training genes; "
            "please align the gene sets before training."
        )
    sc_eval.adata = sc_eval.adata[:, sc_train.adata.var_names]
    sc_eval.ontograph = sc_train.ontograph
    sc_eval.cell_type_index = sc_eval.get_cell_type_index()

    # Validation tissue/species labels must use the class indices established from the training data
    eval_tissues = sc_eval.adata.obs[TISSUE_KEY].astype(str)
    eval_species = sc_eval.adata.obs[SPECIES_KEY].astype(str)
    unseen_tissues = sorted(set(eval_tissues) - set(sc_train.tissue_label_dict))
    unseen_species = sorted(set(eval_species) - set(sc_train.species_label_dict))
    if unseen_tissues or unseen_species:
        raise ValueError(
            "Evaluation data contains labels not seen during training: "
            f"tissue={unseen_tissues}, species={unseen_species}"
        )

    sc_eval.tissue_label_dict = sc_train.tissue_label_dict
    sc_eval.species_label_dict = sc_train.species_label_dict
    sc_eval.tissue_index = np.array(
        [sc_train.tissue_label_dict[value] for value in eval_tissues],
        dtype=np.int64,
    )
    sc_eval.species_index = np.array(
        [sc_train.species_label_dict[value] for value in eval_species],
        dtype=np.int64,
    )

    # ========== 4. Initialize Trainer ==========
    if local_rank == 0:
        print(">>> Initializing UniCell trainer ...")

    trainer = UnicellTrainer(
        sc_train,
        input_type="expr",
        input_dim=input_dim,
        output_dim=512,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        beta=BETA,
        device=device,
        global_layer=GLOBAL_LAYER,
        local_layer=LOCAL_LAYER,
        hidden_layer_dropout=HIDDEN_LAYER_DROPOUT,
        ckpt_dir=CKPT_DIR,
        ddp_train=ddp_train,
        local_rank=local_rank,
    )

    # ========== 5. Train ==========
    if local_rank == 0:
        print(">>> Start training ...")

    trainer.train(scdata_test=sc_eval)

    if local_rank == 0:
        print(">>> Training finished. Checkpoints saved to:", CKPT_DIR)


def main():
    parser = argparse.ArgumentParser()

    # CHANGED: manual train/eval inputs
    parser.add_argument(
        "--train_h5ad",
        type=str,
        required=True,
        help="Training h5ad file."
    )
    parser.add_argument(
        "--eval_h5ad",
        type=str,
        required=True,
        help="Evaluation/validation h5ad file."
    )

    args = parser.parse_args()

    ddp_train, local_rank, device = setup_distributed()

    train_unicell(
        ddp_train=ddp_train,
        local_rank=local_rank,
        device=device,
        train_h5ad=args.train_h5ad,
        eval_h5ad=args.eval_h5ad,
    )

    if ddp_train:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
