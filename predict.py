#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UniCell inference + saving + (optional) test-set Accuracy / Macro-F1 calculation

Key compatibility points for the updated project:
- anno_predict.unicell_predict now supports:
  - 6 outputs from HMCN.forward (celltype/tissue/species)
  - Saving predicted_cell_type(_ontology_id) / predicted_tissue / predicted_species
  - Saving logits to obsm: cls_emb / tissue_cls_emb / species_cls_emb
  - Three-level constrained decoding with safe_list:
      global species/tissue/cell_type whitelist
      species -> tissue
      species + tissue -> cell_type

Example:
python predict.py \
  --input /path/to/test.h5ad \
  --ckpt_dir /path/to/models/Fig5_conservation \
  --output /path/to/results/test_annotated.h5ad \
  --batch_size 2048 \
  --safe_list /path/to/safe_list.json
"""

import os
import argparse
import json
import time
from typing import List, Tuple

import numpy as np
import anndata as ad
import torch

from unicell.anno_predict import unicell_predict
from unicell.utils.utils import compute_metrics as compute_metrics_cls


def _as_str_list(x) -> List[str]:
    return [str(v) for v in list(x)]


def _filter_pairs(
    y_true: List[str],
    y_pred: List[str],
    unknown_tokens: Tuple[str, ...] = ("unknown", "nan", "None", "NA", "N/A", ""),
) -> Tuple[List[str], List[str]]:
    yt, yp = [], []
    for t, p in zip(y_true, y_pred):
        t2 = str(t)
        p2 = str(p)
        if t2 in unknown_tokens or p2 in unknown_tokens:
            continue
        yt.append(t2)
        yp.append(p2)
    return yt, yp


def _eval_task(obs, gt_key: str, pred_key: str, task_name: str):
    if gt_key not in obs.columns:
        print(f"[{task_name}] GT key '{gt_key}' not found in obs -> skip.")
        return None

    if pred_key not in obs.columns:
        print(f"[{task_name}] Pred key '{pred_key}' not found in obs -> skip.")
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


def parse_args():
    p = argparse.ArgumentParser(description="UniCell inference + save + (optional) metrics.")
    p.add_argument("--input", type=str, required=True,
                   help="Input .h5ad (test/eval) file.")
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Checkpoint dir containing unicell_v1.best.pth etc.")
    p.add_argument("--output", type=str,
                   default=None,
                   help="Output annotated .h5ad path. Default: <input>_annotated.h5ad")
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                   help="Device for inference. Default: auto")
    p.add_argument("--batch_size", type=int, default=2048, help="Inference batch size.")
    p.add_argument("--save_metrics_json", action="store_true",
                   help="If set, save metrics as <output>_metrics.json")

    p.add_argument("--cell_type_key", type=str, default="cell_type_ontology_term_id",
                   help="GT cell type key in obs (ontology id).")
    p.add_argument("--tissue_key", type=str, default="general_tissue",
                   help="GT tissue key in obs.")
    p.add_argument("--species_key", type=str, default="organism",
                   help="GT species key in obs.")

    p.add_argument("--safe_list", type=str, default=None,
                   help="Optional path to a hierarchical safe-list JSON file.")

    p.add_argument(
        "--backbone_model",
        default=None,
        help="Override a legacy checkpoint's GeneFormer model directory or scGPT .pt path.",
    )
    p.add_argument(
        "--backbone_vocab",
        default=None,
        help="Override a legacy checkpoint's GeneFormer/scGPT vocabulary JSON.",
    )
    p.add_argument(
        "--backbone_args",
        default=None,
        help="Override a legacy scGPT checkpoint's args.json.",
    )
    p.add_argument(
        "--init_backbone_from_config",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Construct the backbone from config before loading UniCell weights. "
            "Defaults to checkpoint metadata; useful for legacy checkpoint migration."
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.output is None:
        root, ext = os.path.splitext(args.input)
        output_h5ad = root + "_annotated.h5ad"
    else:
        output_h5ad = args.output

    os.makedirs(os.path.dirname(os.path.abspath(output_h5ad)) or ".", exist_ok=True)

    print("=== UniCell Predict ===")
    print(f"Input     : {args.input}")
    print(f"CkptDir   : {args.ckpt_dir}")
    print(f"Device    : {device}")
    print(f"BS        : {args.batch_size}")
    print(f"Output    : {output_h5ad}")
    print(f"SafeList  : {args.safe_list}")
    if any((args.backbone_model, args.backbone_vocab, args.backbone_args)):
        print(f"Backbone  : model={args.backbone_model}")
        print(f"             vocab={args.backbone_vocab}")
        print(f"             args={args.backbone_args}")
    print("=======================")

    t0 = time.perf_counter()
    adata = ad.read_h5ad(args.input)

    sc_dataset = unicell_predict(
        adata=adata,
        filepath=args.input,
        ckpt_dir=args.ckpt_dir,
        batch_size=args.batch_size,
        device=device,
        cell_type_key=args.cell_type_key,
        tissue_key=args.tissue_key,
        species_key=args.species_key,
        compute_metrics=False,
        safe_list_path=args.safe_list,
        backbone_model_file=args.backbone_model,
        backbone_vocab_file=args.backbone_vocab,
        backbone_args_file=args.backbone_args,
        initialize_backbone_from_config=args.init_backbone_from_config,
    )
    t1 = time.perf_counter()
    print(f"[Timing] Inference finished in {t1 - t0:.2f}s")

    t2 = time.perf_counter()
    sc_dataset.adata.write_h5ad(output_h5ad)
    t3 = time.perf_counter()
    print(f"✅ Saved annotated h5ad -> {output_h5ad}")
    print(f"[Timing] write_h5ad finished in {t3 - t2:.2f}s")

    print("\n=== Metrics (if GT available) ===")
    obs = sc_dataset.adata.obs

    metrics = {}

    metrics["cell_type"] = _eval_task(
        obs,
        gt_key=args.cell_type_key,
        pred_key="predicted_cell_type_ontology_id",
        task_name="cell_type(ontology_id)",
    )

    metrics["tissue"] = _eval_task(
        obs,
        gt_key=args.tissue_key,
        pred_key="predicted_tissue",
        task_name="tissue",
    )

    metrics["species"] = _eval_task(
        obs,
        gt_key=args.species_key,
        pred_key="predicted_species",
        task_name="species",
    )

    if args.save_metrics_json:
        metrics_path = os.path.splitext(output_h5ad)[0] + "_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Saved metrics json -> {metrics_path}")

    print("\n=== Saved keys check ===")
    print("obs has:", [k for k in ["predicted_cell_type", "predicted_cell_type_ontology_id",
                               "predicted_tissue", "predicted_species"] if k in obs.columns])
    print("obsm has:", [k for k in ["unicell_emb", "cls_emb", "tissue_cls_emb", "species_cls_emb"]
                        if k in sc_dataset.adata.obsm.keys()])
    print("uns  has:", [k for k in ["cls_cell_type", "cls_tissue", "cls_species"]
                        if k in sc_dataset.adata.uns.keys()])
    print("========================")


if __name__ == "__main__":
    main()
