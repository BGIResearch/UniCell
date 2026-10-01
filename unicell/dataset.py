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


_DEFAULT_SCGPT_PREPROCESS_CHUNK_SIZE = 256


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
        else:
            raise ValueError(f"Unsupported input_type: {self.input_type}")

        self.labels = scDataset.cell_type_index
        self.celltype = scDataset.adata.obs[scDataset.cell_type_key].values
        self.tissue_key = getattr(scDataset, "tissue_key", None)
        self.species_key = getattr(scDataset, "species_key", None)
        self.tissue_index = getattr(scDataset, "tissue_index", None)
        self.species_index = getattr(scDataset, "species_index", None)
        self.batch_key = scDataset.batch_key
        if self.batch_key:
            self.batch = scDataset.batch_index

    def __len__(self):
        if self.input_type == "scGPT":
            return len(self.data["input_ids"])
        else:
            return self.data.shape[0]

    def __getitem__(self, index):
        # expr: return only the index; retrieve data in batches in collate_fn
        if self.input_type == "expr":
            data = index
        elif self.input_type == "scFoundation":
            data = torch.from_numpy(self.data.iloc[index].to_numpy(copy=True))
        elif self.input_type == "GeneFormer":
            data = self.data.select([index])[0]
        elif self.input_type == "scGPT":
            data = {k: v[index] for k, v in self.data.items()}
        else:
            raise ValueError(f"Unsupported input_type: {self.input_type}")

        label = torch.tensor(self.labels[index], dtype=torch.long)
        cls_label = self.celltype[index]

        tissue_label = int(self.tissue_index[index]) if self.tissue_index is not None else -1
        species_label = int(self.species_index[index]) if self.species_index is not None else -1

        if self.batch_key:
            batch_label = torch.tensor(int(self.batch[index]), dtype=torch.long)
            return data, label, cls_label, tissue_label, species_label, batch_label
        else:
            return data, label, cls_label, tissue_label, species_label

    def get_expr_batch(self, indices):
        """
        Retrieve expr data in batches:
        - Slice self.data by indices in a single operation
        - For sparse matrices, call toarray() once for the whole batch
        - Convert the whole batch to torch.float32 in a single operation
        """
        x = self.data[indices]

        if issparse(x):
            x = x.toarray()
        else:
            x = np.asarray(x)

        x = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(x)


def _scgpt_chunk_size(args: Dict, chunk_size: Optional[int]) -> int:
    """Return a validated row chunk size for scGPT preprocessing."""
    if chunk_size is None:
        chunk_size = args.get(
            "preprocess_chunk_size", _DEFAULT_SCGPT_PREPROCESS_CHUNK_SIZE
        )
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size <= 0
    ):
        raise ValueError(
            "scGPT preprocess chunk_size must be a positive integer; "
            f"got {chunk_size!r}."
        )
    return int(chunk_size)


def _max_nonzero_genes_per_cell(expression, chunk_size: int) -> int:
    """Find the longest expressed-gene row without densifying sparse input."""
    max_nonzero = 0
    for start in range(0, expression.shape[0], chunk_size):
        chunk = expression[start : start + chunk_size]
        if issparse(chunk):
            chunk = chunk.tocsr(copy=False)
            # ``getnnz`` counts explicitly stored zeros. They are uncommon, but
            # excluding them keeps the padded width identical to ``np.nonzero``.
            if chunk.nnz and np.any(chunk.data == 0):
                chunk = chunk.copy()
                chunk.eliminate_zeros()
            nonzero_per_cell = np.diff(chunk.indptr)
        else:
            chunk = np.asarray(chunk)
            if chunk.ndim != 2:
                raise ValueError(
                    "scGPT expression data must be a two-dimensional matrix; "
                    f"got shape {chunk.shape}."
                )
            nonzero_per_cell = np.count_nonzero(chunk, axis=1)
        if nonzero_per_cell.size:
            max_nonzero = max(max_nonzero, int(nonzero_per_cell.max()))
    return max_nonzero


def _dense_expression_chunk(expression, start: int, stop: int) -> np.ndarray:
    """Materialize only the requested rows and return a writable array."""
    chunk = expression[start:stop]
    if issparse(chunk):
        return chunk.toarray()
    return np.array(chunk, copy=True)


def load_gpt_data(adata=None, vocab=None, args=None, chunk_size=None):
    """Bin and tokenize scGPT inputs while bounding temporary dense memory.

    Sparse expression data is densified only ``chunk_size`` rows at a time.
    Binning is performed independently for every cell, matching scGPT's
    preprocessing semantics. ``chunk_size`` can be passed explicitly or set as
    ``preprocess_chunk_size`` in ``args``; it defaults to 256 rows.
    """
    chunk_size = _scgpt_chunk_size(args, chunk_size)
    expression = adata.X
    if len(expression.shape) != 2:
        raise ValueError(
            "scGPT expression data must be a two-dimensional matrix; "
            f"got shape {expression.shape}."
        )
    # COO matrices do not support efficient row slicing. Converting to CSR is
    # still sparse and avoids ever allocating the full cells-by-genes array.
    if issparse(expression) and getattr(expression, "format", None) not in {
        "csr",
        "csc",
    }:
        expression = expression.tocsr()

    genes = adata.var_names.tolist()
    pad_token = args["pad_token"]
    vocab.set_default_index(vocab[pad_token])
    gene_ids = np.array(vocab(genes), dtype=int)
    if gene_ids.shape[0] != expression.shape[1]:
        raise ValueError(
            "Number of genes in adata.var_names does not match adata.X: "
            f"{gene_ids.shape[0]} != {expression.shape[1]}."
        )

    max_seq_len = args["max_seq_len"]
    if (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, (int, np.integer))
        or max_seq_len <= 0
    ):
        raise ValueError(
            "scGPT max_seq_len must be a positive integer; "
            f"got {max_seq_len!r}."
        )
    max_nonzero = _max_nonzero_genes_per_cell(expression, chunk_size)
    # One extra position is reserved for the appended <cls> token. Keeping the
    # global width reproduces the shape returned by one-shot tokenization even
    # when an individual chunk contains only short rows.
    token_width = min(max_nonzero + 1, int(max_seq_len))
    n_cells = expression.shape[0]
    input_ids = torch.full(
        (n_cells, token_width), vocab[pad_token], dtype=torch.long
    )
    values = torch.full(
        (n_cells, token_width), args["pad_value"], dtype=torch.float32
    )

    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        counts = _dense_expression_chunk(expression, start, stop)
        for row_index in range(counts.shape[0]):
            row = counts[row_index]
            # Avoid emitting one warning per empty cell while preserving
            # ``binning`` behavior for rows whose maximum is zero.
            if row.max() == 0:
                row.fill(0)
            else:
                row[:] = binning(row, n_bins=args["n_bins"])

        tokenized = tokenize_and_pad_batch(
            counts,
            gene_ids,
            max_len=token_width,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=args["pad_value"],
            append_cls=True,
            include_zero_gene=False,
        )
        chunk_width = tokenized["genes"].shape[1]
        input_ids[start:stop, :chunk_width] = tokenized["genes"]
        values[start:stop, :chunk_width] = tokenized["values"]

    return {"input_ids": input_ids, "values": values}


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
        return x, label

    def __getitem__(self, index):
        if self.txn is None:
            self.env = lmdb.Environment(self.lmdb_path, readonly=True, lock=False)
            self.txn = self.env.begin(write=False)
        exp_x, celltype = self.get_lmdb_data(index)
        data = torch.from_numpy(exp_x)
        label = torch.tensor(celltype)
        return data, label
