#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/29 9:34
# @Author  : qiuping
# @File    : anno_predict.py

"""
UniCell inference / annotation (UPDATED for new project version + MATCH train/eval preprocess + METRICS)

Key upgrades:
1) Preprocess input adata with the SAME logic as scDataset.read_data() used in training/eval:
   - filter_cells(min_genes=200) if n_vars>1000
   - normalize_total(target_sum=1e4) when looks like integer counts and max>25
   - log1p when max>25
2) Compatible with new HMCN:
   - HMCN.forward returns 6 outputs:
       (global_layer_activation, global_layer_output, local_layer_outputs,
        global_cls_output, tissue_cls_output, species_cls_output)
3) Save into AnnData:
   - celltype: predicted_cell_type(_ontology_id), cls_emb, cls_cell_type
   - tissue/species: predicted_tissue/predicted_species, tissue_cls_emb/species_cls_emb, cls_tissue/cls_species
   - embedding: unicell_emb
   - hierarchy path: level_0..level_k based on predicted cell type ontology id
4) OPTIONAL: compute & print accuracy/macro_f1 if GT columns exist in adata.obs:
   - cell_type_key (default: cell_type_ontology_term_id)
   - tissue_key    (default: general_tissue)
   - species_key   (default: organism)
5) OPTIONAL: hierarchical constrained decoding by safe_list JSON:
   - global species/tissue/cell_type whitelist
   - species -> tissue
   - species + tissue -> cell_type
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

import torch
import anndata as ad
import networkx as nx
from scipy import sparse
from torch.utils.data import DataLoader

import scanpy as sc

from unicell.hmcn import HMCN
from unicell.utils.utils import load_ontograph
from unicell.utils.utils import compute_metrics as compute_metrics_cls
from unicell.scDataset import scDataset
from unicell.dataset import HMCNDataset
from unicell.repo.geneformer.in_silico_perturber import get_model_input_size, pad_tensor_list


# ----------------------------
# 1) Match train/eval preprocess
# ----------------------------
def _match_train_eval_preprocess(adata: ad.AnnData) -> ad.AnnData:
    """
    Copy of scDataset.read_data() preprocessing logic (for consistency):
    - if X.min() >= 0:
        if n_vars > 1000: filter_cells(min_genes=200)
        if X.max() > 25:
            log1p = True
            if X.max() is integer-like: normalize_total = 1e4
        if normalize_total: sc.pp.normalize_total(target_sum=1e4)
        if log1p: sc.pp.log1p
    """
    try:
        x_min = adata.X.min()
    except Exception:
        x_min = 0

    if x_min >= 0:
        if adata.n_vars > 1000:
            sc.pp.filter_cells(adata, min_genes=200)

        normalize_total = False
        log1p = False

        try:
            x_max = adata.X.max()
        except Exception:
            x_max = None

        if x_max is not None and x_max > 25:
            log1p = True
            try:
                if x_max - np.int32(x_max) == np.int32(0):
                    normalize_total = 1e4
            except Exception:
                pass

        if normalize_total:
            sc.pp.normalize_total(adata, target_sum=normalize_total)
            print("[Preprocess] normalize_total(target_sum=1e4)")

        if log1p:
            sc.pp.log1p(adata)
            print("[Preprocess] log1p")

    return adata


# ----------------------------
# 2) Metrics helpers
# ----------------------------
def _as_str_list(x):
    return [str(v) for v in list(x)]


def _filter_pairs(y_true, y_pred, unknown_tokens=("unknown", "nan", "None", "NA", "N/A", "")):
    yt, yp = [], []
    for t, p in zip(y_true, y_pred):
        t2, p2 = str(t), str(p)
        if t2 in unknown_tokens or p2 in unknown_tokens:
            continue
        yt.append(t2)
        yp.append(p2)
    return yt, yp


def _eval_task(obs: pd.DataFrame, gt_key: str, pred_key: str, task_name: str):
    if gt_key not in obs.columns:
        print(f"[{task_name}] GT key '{gt_key}' not found -> skip.")
        return None
    if pred_key not in obs.columns:
        print(f"[{task_name}] Pred key '{pred_key}' not found -> skip.")
        return None

    y_true = _as_str_list(obs[gt_key].values)
    y_pred = _as_str_list(obs[pred_key].values)
    y_true, y_pred = _filter_pairs(y_true, y_pred)

    if len(y_true) == 0:
        print(f"[{task_name}] No valid pairs after filtering -> skip.")
        return None

    res = compute_metrics_cls(y_true, y_pred)
    acc = float(res.get("accuracy", np.nan))
    mf1 = float(res.get("macro_f1", np.nan))
    print(f"[{task_name}] n={len(y_true)}  accuracy={acc:.4f}  macro_f1={mf1:.4f}")
    return {"n": int(len(y_true)), "accuracy": acc, "macro_f1": mf1}


# ----------------------------
# 3) Safe-list helpers
# ----------------------------
def _load_safe_list(safe_list_path):
    if safe_list_path is None:
        return None
    with open(safe_list_path, "r", encoding="utf-8") as f:
        safe_list = json.load(f)
    print(f"[SafeList] loaded from: {safe_list_path}")
    return safe_list


def _build_allowed_index_set(names, label_dict, field_name="unknown"):
    """
    names: list[str]
    label_dict: {name: idx}
    return: set[int]
    """
    allowed = set()
    missing = []
    for x in names:
        if x in label_dict:
            allowed.add(label_dict[x])
        else:
            missing.append(x)

    if len(missing) > 0:
        print(f"[SafeList][WARN] {field_name} missing labels (first 20): {missing[:20]}")
    return allowed


def _parse_safe_constraints(safe_list, celltype_label_dict, tissue_dict, species_dict):
    """
    Returns:
      global_species_allowed: set[int] or None
      global_tissue_allowed: set[int] or None
      global_celltype_allowed: set[int] or None
      species_to_tissue_allowed: dict[int, set[int]]
      species_tissue_to_celltype_allowed: dict[(int, int), set[int]]
    """
    if safe_list is None:
        return None, None, None, {}, {}

    global_cfg = safe_list.get("global", {})
    hierarchy_cfg = safe_list.get("hierarchy", {})

    global_species_allowed = None
    global_tissue_allowed = None
    global_celltype_allowed = None

    if "species" in global_cfg:
        global_species_allowed = _build_allowed_index_set(
            global_cfg["species"], species_dict, "global.species"
        )
    if "tissue" in global_cfg:
        global_tissue_allowed = _build_allowed_index_set(
            global_cfg["tissue"], tissue_dict, "global.tissue"
        )
    if "cell_type" in global_cfg:
        global_celltype_allowed = _build_allowed_index_set(
            global_cfg["cell_type"], celltype_label_dict, "global.cell_type"
        )

    species_to_tissue_allowed = {}
    species_tissue_to_celltype_allowed = {}

    for species_name, species_block in hierarchy_cfg.items():
        if species_name not in species_dict:
            print(f"[SafeList][WARN] hierarchy species not found: {species_name}")
            continue

        species_idx = species_dict[species_name]
        tissue_block = species_block.get("tissue", {})

        allowed_tissue_set = set()

        for tissue_name, tissue_cfg in tissue_block.items():
            if tissue_name not in tissue_dict:
                print(f"[SafeList][WARN] hierarchy tissue not found: {(species_name, tissue_name)}")
                continue

            tissue_idx = tissue_dict[tissue_name]
            allowed_tissue_set.add(tissue_idx)

            ct_names = tissue_cfg.get("cell_type", None)
            if ct_names is not None:
                ct_allowed = _build_allowed_index_set(
                    ct_names, celltype_label_dict, f"hierarchy.{species_name}.tissue.{tissue_name}.cell_type"
                )
                species_tissue_to_celltype_allowed[(species_idx, tissue_idx)] = ct_allowed

        if len(allowed_tissue_set) > 0:
            species_to_tissue_allowed[species_idx] = allowed_tissue_set

    return (
        global_species_allowed,
        global_tissue_allowed,
        global_celltype_allowed,
        species_to_tissue_allowed,
        species_tissue_to_celltype_allowed,
    )


def _masked_argmax_1d(logit_row, allowed_idx_set):
    """
    logit_row: np.ndarray shape [C]
    allowed_idx_set: set[int] or None
    """
    if allowed_idx_set is None:
        return int(np.argmax(logit_row))

    if len(allowed_idx_set) == 0:
        return int(np.argmax(logit_row))

    masked = np.full_like(logit_row, -1e9, dtype=np.float32)
    allowed_idx = np.array(sorted(list(allowed_idx_set)), dtype=np.int64)
    masked[allowed_idx] = logit_row[allowed_idx]
    return int(np.argmax(masked))


def _hierarchical_constrained_decode(
    cls_logits,
    tissue_logits,
    species_logits,
    celltype_label_dict,
    tissue_dict,
    species_dict,
    safe_list,
):
    """
    Three-level coupled decoding:
      species -> tissue -> cell_type

    Inputs:
      cls_logits:     [N, C_cell]
      tissue_logits:  [N, C_tissue]
      species_logits: [N, C_species]

    Returns:
      cell_pred_idx, tissue_pred_idx, species_pred_idx
    """
    (
        global_species_allowed,
        global_tissue_allowed,
        global_celltype_allowed,
        species_to_tissue_allowed,
        species_tissue_to_celltype_allowed,
    ) = _parse_safe_constraints(
        safe_list=safe_list,
        celltype_label_dict=celltype_label_dict,
        tissue_dict=tissue_dict,
        species_dict=species_dict,
    )

    n = cls_logits.shape[0]
    cell_pred_idx = np.zeros(n, dtype=np.int64)
    tissue_pred_idx = np.zeros(n, dtype=np.int64)
    species_pred_idx = np.zeros(n, dtype=np.int64)

    for i in range(n):
        # 1) species
        sp_idx = _masked_argmax_1d(species_logits[i], global_species_allowed)
        species_pred_idx[i] = sp_idx

        # 2) tissue = global_tissue ∩ species_to_tissue[sp]
        tissue_allowed = None
        if global_tissue_allowed is not None:
            tissue_allowed = set(global_tissue_allowed)

        if sp_idx in species_to_tissue_allowed:
            sp_tissue_allowed = species_to_tissue_allowed[sp_idx]
            tissue_allowed = sp_tissue_allowed if tissue_allowed is None else (tissue_allowed & sp_tissue_allowed)

        ti_idx = _masked_argmax_1d(tissue_logits[i], tissue_allowed)
        tissue_pred_idx[i] = ti_idx

        # 3) celltype = global_celltype ∩ species_tissue_to_celltype[(sp,ti)]
        cell_allowed = None
        if global_celltype_allowed is not None:
            cell_allowed = set(global_celltype_allowed)

        key = (sp_idx, ti_idx)
        if key in species_tissue_to_celltype_allowed:
            st_cell_allowed = species_tissue_to_celltype_allowed[key]
            cell_allowed = st_cell_allowed if cell_allowed is None else (cell_allowed & st_cell_allowed)

        ct_idx = _masked_argmax_1d(cls_logits[i], cell_allowed)
        cell_pred_idx[i] = ct_idx

    return cell_pred_idx, tissue_pred_idx, species_pred_idx


# ----------------------------
# 4) Main prediction
# ----------------------------
def unicell_predict(
    adata,
    filepath,
    ckpt_dir=None,
    batch_size=512,
    device=None,
    cell_type_key: str = "cell_type_ontology_term_id",
    tissue_key: str = "general_tissue",
    species_key: str = "organism",
    compute_metrics: bool = True,
    safe_list_path: str = None,
    backbone_model_file: str = None,
    backbone_vocab_file: str = None,
    backbone_args_file: str = None,
    initialize_backbone_from_config: bool = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    safe_list = _load_safe_list(safe_list_path)

    # preprocess
    adata = _match_train_eval_preprocess(adata.copy())

    model_path = os.path.join(ckpt_dir, "unicell_v1.best.pth")
    model_dict = torch.load(model_path, map_location="cpu")
    metadata = dict(model_dict["metadata"])

    # A portable foundation-model checkpoint stores lightweight reconstruction
    # assets relative to ckpt_dir. Explicit caller overrides remain available
    # for legacy checkpoints whose metadata points to another machine.
    overrides = {
        "llm_model_file": backbone_model_file,
        "llm_vocab_file": backbone_vocab_file,
        "llm_args_file": backbone_args_file,
    }
    for key, value in overrides.items():
        if value is not None:
            metadata[key] = value

    for key in ("llm_model_file", "llm_vocab_file", "llm_args_file"):
        value = metadata.get(key)
        if value and not os.path.isabs(value):
            metadata[key] = os.path.abspath(os.path.join(ckpt_dir, value))

    if initialize_backbone_from_config is None:
        initialize_backbone_from_config = bool(
            metadata.get("backbone_init_from_config", False)
        )

    with open(os.path.join(ckpt_dir, "celltype_dict.pk"), "rb") as f:
        celltype_label_dict = pickle.load(f)
    with open(os.path.join(ckpt_dir, "tissue_dict.pk"), "rb") as f:
        tissue_dict = pickle.load(f)
    with open(os.path.join(ckpt_dir, "species_dict.pk"), "rb") as f:
        species_dict = pickle.load(f)

    tissue_cls_num = len(tissue_dict)
    species_cls_num = len(species_dict)

    sc_dataset = read_data(
        adata=adata,
        filepath=filepath,
        ckpt_dir=ckpt_dir,
        llm_vocab=metadata.get("llm_vocab_file", None),
        llm_args=metadata.get("llm_args_file", None),
        tissue_dict=tissue_dict,
        species_dict=species_dict,
        tissue_key=tissue_key,
        species_key=species_key,
    )

    hierarchical_array = sc_dataset.ontograph.hierarchical_array
    num_classes = len(hierarchical_array)
    print("num_class:", num_classes)

    hierarchical_class = [arr.shape[1] for arr in hierarchical_array]
    hierarchical_depth = [metadata["global_layer"] if i > 0 else 0 for i in range(num_classes)]
    global2local = [metadata["local_layer"] if i > 0 else 0 for i in range(num_classes)]

    cls2id = {v: sc_dataset.ontograph.vocab[k] for k, v in celltype_label_dict.items()}

    model = HMCN(
        input_type=metadata["input_type"],
        input_dim=metadata["input_dim"],
        output_dim=metadata["output_dim"],
        num_classes=num_classes,
        hierarchical_depth=hierarchical_depth,
        global2local=global2local,
        hierarchical_class=hierarchical_class,
        hidden_layer_dropout=metadata["hidden_layer_dropout"],
        cls_num=len(celltype_label_dict),
        tissue_cls_num=tissue_cls_num,
        species_cls_num=species_cls_num,
        llm_model_file=metadata["llm_model_file"],
        llm_vocab_file=metadata["llm_vocab_file"],
        llm_args_file=metadata["llm_args_file"],
        initialize_backbone_from_config=initialize_backbone_from_config,
    )

    target_state = model.state_dict()
    params = {}
    for k, v in model_dict["model_state_dict"].items():
        normalized_key = k.removeprefix("module.")
        candidate_keys = [normalized_key]
        if ".self_attn.Wqkv.weight" in normalized_key:
            candidate_keys.append(
                normalized_key.replace(
                    ".self_attn.Wqkv.weight", ".self_attn.in_proj_weight"
                )
            )
        elif ".self_attn.Wqkv.bias" in normalized_key:
            candidate_keys.append(
                normalized_key.replace(
                    ".self_attn.Wqkv.bias", ".self_attn.in_proj_bias"
                )
            )
        target_key = next(
            (
                candidate
                for candidate in candidate_keys
                if candidate in target_state and target_state[candidate].shape == v.shape
            ),
            normalized_key,
        )
        params[target_key] = v
    model.load_state_dict(params, strict=True)
    model = model.eval().to(device)

    # prepare dataset for inference
    sc_dataset.cell_type_index = [0] * len(sc_dataset.adata)
    sc_dataset.cell_type_key = "cell_type_pseudo"
    sc_dataset.adata.obs[sc_dataset.cell_type_key] = [0] * len(sc_dataset.adata)

    dataset = HMCNDataset(sc_dataset, input_type=metadata["input_type"])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_with_args(
            input_type=metadata["input_type"],
            model=model,
            dataset=dataset,
        ),
        num_workers=4,
        pin_memory=True,
    )

    cls_outs, tissue_outs, species_outs, cell_embs = [], [], [], []

    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            if len(batch) == 5:
                batch_data, batch_labels, cls_labels, tissue_labels, species_labels = batch
            else:
                batch_data, batch_labels, cls_labels, tissue_labels, species_labels, batch_batch_labels = batch

            if metadata["input_type"] == "GeneFormer":
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                model.geneformer = model.geneformer.to(device)
            elif metadata["input_type"] == "scGPT":
                batch_data = {
                    "input_ids": torch.stack([b["input_ids"] for b in batch_data]).to(torch.long).to(device),
                    "values": torch.stack([b["values"] for b in batch_data]).to(device),
                }
            elif metadata["input_type"] == "expr":
                batch_data = batch_data.to(device, non_blocking=True)
            else:
                batch_data = torch.stack(batch_data).to(device)

            with torch.cuda.amp.autocast(enabled=(metadata["input_type"] == "scGPT")):
                encoder_x, global_out, local_out, cls_out, tissue_out, species_out = model(batch_data)

            cls_outs.append(cls_out)
            tissue_outs.append(tissue_out)
            species_outs.append(species_out)
            cell_embs.append(encoder_x.detach().cpu().numpy())

    cls_layer_output = torch.cat(cls_outs, dim=0).detach().cpu().numpy()
    tissue_logits = torch.cat(tissue_outs, dim=0).detach().cpu().numpy()
    species_logits = torch.cat(species_outs, dim=0).detach().cpu().numpy()

    if safe_list is None:
        cls_pred_idx = np.argmax(cls_layer_output, axis=1)
        tissue_pred_idx = np.argmax(tissue_logits, axis=1)
        species_pred_idx = np.argmax(species_logits, axis=1)
    else:
        cls_pred_idx, tissue_pred_idx, species_pred_idx = _hierarchical_constrained_decode(
            cls_logits=cls_layer_output,
            tissue_logits=tissue_logits,
            species_logits=species_logits,
            celltype_label_dict=celltype_label_dict,
            tissue_dict=tissue_dict,
            species_dict=species_dict,
            safe_list=safe_list,
        )

    labels_pred = [cls2id[idx] for idx in cls_pred_idx]
    vocab = {v: k for k, v in sc_dataset.ontograph.vocab.items()}

    sc_dataset.adata.obs["predicted_cell_type_ontology_id"] = [vocab[i] for i in labels_pred]
    sc_dataset.adata.obs["predicted_cell_type"] = [
        str(i) for i in sc_dataset.adata.obs["predicted_cell_type_ontology_id"]
    ]

    sc_dataset.adata.obsm["unicell_emb"] = np.concatenate(cell_embs, axis=0)
    sc_dataset.adata.obsm["cls_emb"] = cls_layer_output
    sc_dataset.adata.uns["cls_cell_type"] = [
        str(vocab[cls2id[i]])
        for i in range(len(cls2id))
    ]

    idx2tissue = {v: k for k, v in tissue_dict.items()}
    idx2species = {v: k for k, v in species_dict.items()}

    sc_dataset.adata.obs["predicted_tissue"] = [idx2tissue[i] for i in tissue_pred_idx]
    sc_dataset.adata.obs["predicted_species"] = [idx2species[i] for i in species_pred_idx]

    sc_dataset.adata.obsm["tissue_cls_emb"] = tissue_logits
    sc_dataset.adata.obsm["species_cls_emb"] = species_logits
    sc_dataset.adata.uns["cls_tissue"] = [idx2tissue[i] for i in range(tissue_cls_num)]
    sc_dataset.adata.uns["cls_species"] = [idx2species[i] for i in range(species_cls_num)]

    celltype_preds = list(set(sc_dataset.adata.obs["predicted_cell_type_ontology_id"]))
    hierarchical_level = ["level_" + str(i) for i in range(max(sc_dataset.ontograph.hierarchical_levels.values()) + 1)]
    hierarchical_level_df = pd.DataFrame(index=celltype_preds, columns=hierarchical_level)

    lca = sc_dataset.ontograph.common_ancestor
    for i, celltype in enumerate(celltype_preds):
        path = nx.shortest_path(sc_dataset.ontograph.graph, lca, celltype)
        hierarchical_level_df.iloc[i, : len(path)] = [str(node) for node in path]

    # Store ontology levels as categoricals so shorter paths retain genuine
    # missing values while AnnData/HDF5 can serialize even all-missing levels.
    # This only changes the storage dtype of level_* columns; predictions,
    # embeddings, logits, and metric inputs above remain unchanged.
    for column in hierarchical_level_df.columns:
        hierarchical_level_df[column] = hierarchical_level_df[column].astype("category")

    hierarchical_level_df = hierarchical_level_df.loc[sc_dataset.adata.obs["predicted_cell_type_ontology_id"]]
    hierarchical_level_df.index = sc_dataset.adata.obs_names
    sc_dataset.adata.obs = pd.concat([sc_dataset.adata.obs, hierarchical_level_df], axis=1)

    if compute_metrics:
        print("\n=== Metrics (anno_predict) ===")
        obs = sc_dataset.adata.obs

        _eval_task(
            obs,
            gt_key=cell_type_key,
            pred_key="predicted_cell_type_ontology_id",
            task_name="cell_type(ontology_id)",
        )
        _eval_task(
            obs,
            gt_key=tissue_key,
            pred_key="predicted_tissue",
            task_name="tissue",
        )
        _eval_task(
            obs,
            gt_key=species_key,
            pred_key="predicted_species",
            task_name="species",
        )
        print("=============================\n")

    return sc_dataset


def read_data(
    adata,
    filepath,
    ckpt_dir,
    llm_vocab,
    llm_args,
    tissue_dict=None,
    species_dict=None,
    tissue_key="general_tissue",
    species_key="organism",
):
    """
    - Build scDataset(trained=False) but from preprocessed adata
    - Load ontograph
    - Align genes to training gene order
    - Prepare tissue/species indices if columns exist
    """
    scdataset = scDataset(
        adata=adata,
        data_path=filepath,
        cell_type_key=None,
        trained=False,
        highly_variable_genes=False,
        llm_vocab=llm_vocab,
        llm_args=llm_args,
    )

    gene_path = os.path.join(ckpt_dir, "gene_names.pk")
    with open(gene_path, "rb") as f:
        gene_names = pickle.load(f)

    scdataset.ontograph = load_ontograph(ckpt_dir)

    new_data = np.zeros((scdataset.adata.X.shape[0], len(gene_names)), dtype=np.float32)
    useful_gene_index = np.where(scdataset.adata.var_names.isin(gene_names))
    useful_gene = scdataset.adata.var_names[useful_gene_index]
    if len(useful_gene) == 0:
        raise ValueError("No gene names in ref gene. Please check adata.var_names are gene symbols.")
    print("useful gene index:", len(useful_gene))

    gene_names = list(gene_names)
    gene_index = [gene_names.index(i) for i in useful_gene]

    if not sparse.issparse(scdataset.adata.X):
        new_data[:, gene_index] = scdataset.adata[:, useful_gene_index[0]].X
    else:
        new_data[:, gene_index] = scdataset.adata[:, useful_gene_index[0]].X.toarray()

    new_data = sparse.csr_matrix(new_data)
    new_adata = ad.AnnData(
        X=new_data,
        obs=scdataset.adata.obs,
        obsm=scdataset.adata.obsm,
        uns=scdataset.adata.uns,
    )
    new_adata.var_names = gene_names
    scdataset.adata = new_adata

    if tissue_dict is not None and tissue_key in scdataset.adata.obs.columns:
        scdataset.tissue_key = tissue_key
        vals = scdataset.adata.obs[tissue_key].astype(str).fillna("unknown")
        scdataset.adata.obs[tissue_key] = vals
        scdataset.tissue_label_dict = tissue_dict
        scdataset.tissue_index = np.array([tissue_dict.get(v, -1) for v in vals.tolist()], dtype=np.int64)

    if species_dict is not None and species_key in scdataset.adata.obs.columns:
        scdataset.species_key = species_key
        vals = scdataset.adata.obs[species_key].astype(str).fillna("unknown")
        scdataset.adata.obs[species_key] = vals
        scdataset.species_label_dict = species_dict
        scdataset.species_index = np.array([species_dict.get(v, -1) for v in vals.tolist()], dtype=np.int64)

    return scdataset


def collate_fn_with_args(input_type, model, dataset):
    """
    Collate function compatible with new HMCNDataset returning 5/6 fields.

    - GeneFormer: pad input_ids
    - expr: batch_data contains indices; call dataset.get_expr_batch(indices)
    - other: pass-through
    """
    def collate_fn(batch):
        if len(batch[0]) == 5:
            batch_data, batch_labels, cls_labels, tissue_labels, species_labels = zip(*batch)
            batch_batch_labels = None
        else:
            batch_data, batch_labels, cls_labels, tissue_labels, species_labels, batch_batch_labels = zip(*batch)

        batch_labels = torch.stack(batch_labels)

        if input_type == "GeneFormer":
            model_input_size = get_model_input_size(model.geneformer)

            max_len = max(data["length"] for data in batch_data)
            input_data_minibatch = [torch.tensor(data["input_ids"], dtype=torch.long) for data in batch_data]
            pad_token_id = model.llm_vocab["<pad>"]
            input_data_minibatch = pad_tensor_list(input_data_minibatch, max_len, pad_token_id, model_input_size)

            new_batch_data = {
                "input_ids": input_data_minibatch,
                "length": torch.tensor([data["length"] for data in batch_data], dtype=torch.long),
            }

            if batch_batch_labels is None:
                return new_batch_data, batch_labels, cls_labels, list(tissue_labels), list(species_labels)
            else:
                return new_batch_data, batch_labels, cls_labels, list(tissue_labels), list(species_labels), torch.stack(batch_batch_labels)

        elif input_type == "expr":
            indices = list(batch_data)
            expr_batch = dataset.get_expr_batch(indices)

            if batch_batch_labels is None:
                return expr_batch, batch_labels, cls_labels, list(tissue_labels), list(species_labels)
            else:
                return expr_batch, batch_labels, cls_labels, list(tissue_labels), list(species_labels), torch.stack(batch_batch_labels)

        else:
            if batch_batch_labels is None:
                return batch_data, batch_labels, cls_labels, list(tissue_labels), list(species_labels)
            else:
                return batch_data, batch_labels, cls_labels, list(tissue_labels), list(species_labels), torch.stack(batch_batch_labels)

    return collate_fn
