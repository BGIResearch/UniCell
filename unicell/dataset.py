#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataset.py
# @Time      :2024/4/11 20:38
# @Author    :Luni Hu

import torch
from torch.utils.data import Dataset
import numpy as np
import lmdb
import json
from scipy.sparse import issparse
from typing import Dict, Optional, Union

import pandas as pd
from unicell.repo.scfoundation.get_embedding import main_gene_selection
from unicell.repo.geneformer.tokenizer import TranscriptomeTokenizer
from unicell.repo.scgpt.tokenizer.gene_tokenizer import tokenize_and_pad_batch

def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits

def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
    # TODO: use torch.quantile and torch.bucketize

    if row.max() == 0:
        print(
            "The input data contains row of zeros. Please make sure this is expected."
        )
        return (
            np.zeros_like(row, dtype=dtype)
            if return_np
            else torch.zeros_like(row, dtype=dtype)
        )

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)


class HMCNDataset(Dataset):
    def __init__(self, scDataset, input_type):
        self.input_type = input_type
        if self.input_type == "expr":
            self.data = scDataset.adata.X
        elif self.input_type == "scFoundation":
            self.data = load_scf_data(scDataset.adata, scDataset.llm_vocab)
        elif self.input_type == "GeneFormer":
            self.data = load_gf_data(scDataset.adata)
        elif self.input_type == "scGPT":
            self.data = load_gpt_data(scDataset.adata, scDataset.llm_vocab, scDataset.llm_args)

        self.labels = scDataset.cell_type_index
        self.celltype = scDataset.adata.obs[scDataset.cell_type_key].values
        self.batch_key = scDataset.batch_key
        if self.batch_key:
            self.batch = scDataset.batch_index

    def __len__(self):

        if self.input_type == "scGPT":
            return len(self.data["input_ids"])
        else:
            return self.data.shape[0]

    def __getitem__(self, index):
        data=None

        if self.input_type == "expr":
            sample = self.data[index].A[0] if issparse(self.data[index]) else self.data[index]
        elif self.input_type == "scFoundation":
            sample = self.data.iloc[index]
        elif self.input_type == "GeneFormer":
            data = self.data.select([index])[0]
        elif self.input_type == "scGPT":
            data = {k:v[index] for k, v in self.data.items()}

        label = self.labels[index]

        if not self.input_type in ["GeneFormer", "scGPT"]:
            data = torch.tensor(sample)
            data = data.to(torch.float32)
        label = torch.tensor(label)
        cls_label = self.celltype[index]
        if self.batch_key:
            batch_label = self.batch[index]
            batch_label = torch.tensor(batch_label, dtype=torch.long)
            return data, label, cls_label, batch_label
        else:
            return data, label, cls_label



def load_gpt_data(adata=None, vocab=None, args=None):

    all_counts = (
        binning(adata.X.toarray() if issparse(adata.X) else adata.X, n_bins=args["n_bins"])
    )
    genes = adata.var_names.tolist()
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    tokenized_datasets = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=args["max_seq_len"],
        vocab=vocab,
        pad_token=args["pad_token"],
        pad_value=args["pad_value"],
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=False
    )

    input_gene_ids_train = tokenized_datasets["genes"]
    target_values_train = tokenized_datasets["values"]

    data_pt = {"input_ids": input_gene_ids_train,
                "values": target_values_train
        }

    return data_pt

def load_scf_data(adata=None, vocab=None):
    idx = adata.obs_names.tolist()
    col = adata.var_names.tolist()
    if issparse(adata.X):
        gexpr_feature = adata.X.toarray()
    else:
        gexpr_feature = adata.X
    gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)
    gene_list = list(vocab.get_stoi().keys())
    gexpr_feature = gexpr_feature.loc[:, gexpr_feature.columns.isin(gene_list)]
    gexpr_feature, to_fill_columns, var = main_gene_selection(gexpr_feature, gene_list)
    assert gexpr_feature.shape[1] == 19264
    return gexpr_feature

def load_gf_data(adata=None, nproc=16):
    tk = TranscriptomeTokenizer(nproc=nproc)
    tokenized_cells, cell_metadata = tk.tokenize_anndata(adata)
    tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)
    return tokenized_dataset

class HMCNDatasetLmdb(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = lmdb.Environment(self.lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin(write=False)
        self.length = self.get_length()
        self.env = None
        self.txn = None

    def get_length(self):
        length = int(self.txn.get(b'__len__').decode())
        print('lmdb length: ', length)
        return length

    def __len__(self):
        return self.length

    def get_lmdb_data(self, index):
        value = json.loads(self.txn.get(str(index).encode()))
        x = value['x']
        label = value['label']
        # gene = lmdb value['gene']
        return x, label

    def __getitem__(self, index):
        if self.txn is None:
            self.env = lmdb.Environment(self.lmdb_path, readonly=True, lock=False)
            self.txn = self.env.begin(write=False)
        exp_x, celltype = self.get_lmdb_data(index)
        data = torch.from_numpy(exp_x)
        label = torch.tensor(celltype)
        return data, label




