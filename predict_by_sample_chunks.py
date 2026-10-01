#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run UniCell inference in chunks grouped by obs[sample] to avoid running out of memory when loading the entire file.

Features:
1. Open the large h5ad file with backed='r' only once throughout the script
2. Group by sample and load only one sample subset into memory at a time
3. Call the existing unicell_predict function for inference
4. Save each sample as a separate h5ad file
5. Optionally save metrics as JSON for each chunk
6. Optionally skip completed chunks to resume interrupted runs
7. Also save predictions as a CSV file for each sample
"""

import os
import re
import json
import time
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import torch

from unicell.anno_predict import unicell_predict
from unicell.utils.utils import compute_metrics as compute_metrics_cls


# -------------------------
# helpers
# -------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _sanitize_filename(s: str, max_len: int = 120) -> str:
    s = str(s)
    s = s.strip()
    s = re.sub(r"[\\/:*?\"<>|\s]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "EMPTY"
    return s[:max_len]


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


def _compute_metrics_for_adata(
    adata,
    cell_type_key: str,
    tissue_key: str,
    species_key: str,
):
    obs = adata.obs
    metrics = {}

    metrics["cell_type"] = _eval_task(
        obs,
        gt_key=cell_type_key,
        pred_key="predicted_cell_type_ontology_id",
        task_name="cell_type(ontology_id)",
    )
    metrics["tissue"] = _eval_task(
        obs,
        gt_key=tissue_key,
        pred_key="predicted_tissue",
        task_name="tissue",
    )
    metrics["species"] = _eval_task(
        obs,
        gt_key=species_key,
        pred_key="predicted_species",
        task_name="species",
    )
    return metrics


def _save_prediction_csv(adata, sample_key: str, sample_value: str, out_csv: str):
    need_cols = [
        "predicted_cell_type_ontology_id",
        "predicted_tissue",
        "predicted_species",
    ]
    missing = [c for c in need_cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing prediction columns in adata.obs: {missing}")

    df = pd.DataFrame({
        "obs_name": adata.obs_names.astype(str),
        "predicted_cell_type_ontology_id": adata.obs["predicted_cell_type_ontology_id"].astype(str).values,
        "predicted_tissue": adata.obs["predicted_tissue"].astype(str).values,
        "predicted_species": adata.obs["predicted_species"].astype(str).values,
    })

    if sample_key in adata.obs.columns:
        df["sample"] = adata.obs[sample_key].astype(str).values
    else:
        df["sample"] = str(sample_value)

    df.to_csv(out_csv, index=False)
    print(f"[Done] prediction csv -> {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="Chunked UniCell inference by obs[sample].")
    p.add_argument("--input", type=str, required=True, help="Input large .h5ad")
    p.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint dir")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to save per-sample outputs")

    p.add_argument("--sample_key", type=str, default="sample", help="obs column used for chunk split")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    p.add_argument("--batch_size", type=int, default=2048, help="Inference batch size")
    p.add_argument("--safe_list", type=str, default=None, help="Optional path to a safe-list JSON file")
    p.add_argument("--compression", type=str, default="gzip", choices=["gzip", "none"], help="Output compression")
    p.add_argument("--compression_opts", type=int, default=4, help="gzip compression level")
    p.add_argument("--skip_done", action="store_true", help="Skip already finished sample chunk")
    p.add_argument("--save_metrics_json", action="store_true", help="Save metrics json for each chunk")
    p.add_argument("--summary_csv", type=str, default=None, help="Optional csv path for chunk summary")

    p.add_argument("--cell_type_key", type=str, default="cell_type_ontology_term_id")
    p.add_argument("--tissue_key", type=str, default="general_tissue")
    p.add_argument("--species_key", type=str, default="organism")
    p.add_argument("--backbone_model", default=None, help="Override a legacy backbone model path")
    p.add_argument("--backbone_vocab", default=None, help="Override a legacy backbone vocab path")
    p.add_argument("--backbone_args", default=None, help="Override a legacy scGPT args path")
    p.add_argument(
        "--init_backbone_from_config",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Construct backbone from config before loading the UniCell state dict",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU.")
        device = "cpu"
    else:
        device = args.device

    _ensure_dir(args.out_dir)

    print("=== Chunked UniCell Predict by Sample ===")
    print(f"Input        : {args.input}")
    print(f"CkptDir      : {args.ckpt_dir}")
    print(f"OutDir       : {args.out_dir}")
    print(f"SampleKey    : {args.sample_key}")
    print(f"Device       : {device}")
    print(f"BatchSize    : {args.batch_size}")
    print(f"SafeList     : {args.safe_list}")
    print(f"SkipDone     : {args.skip_done}")
    print("=========================================")

    t0 = time.perf_counter()
    adata_b = None
    summary_rows = []

    try:
        print(f"[Step1] Open backed h5ad once: {args.input}")
        adata_b = ad.read_h5ad(args.input, backed="r")

        if args.sample_key not in adata_b.obs.columns:
            raise KeyError(f"sample_key '{args.sample_key}' not found in obs columns.")

        obs_names = np.array(adata_b.obs_names.astype(str))
        sample_series = adata_b.obs[args.sample_key].astype(str).copy()

        unique_samples = pd.Index(sample_series).drop_duplicates().tolist()
        print(f"[Info] total cells   : {len(obs_names)}")
        print(f"[Info] total samples : {len(unique_samples)}")

        for i, sample_value in enumerate(unique_samples, start=1):
            sample_mask = (sample_series.values == sample_value)
            cell_pos = np.where(sample_mask)[0]   # Use integer positional indices instead of cell_ids
            n_cells = int(cell_pos.size)

            sample_tag = _sanitize_filename(sample_value)
            out_h5ad = os.path.join(args.out_dir, f"pred_{i:06d}_{sample_tag}.h5ad")
            out_csv = os.path.join(args.out_dir, f"pred_{i:06d}_{sample_tag}.csv")
            out_json = os.path.join(args.out_dir, f"pred_{i:06d}_{sample_tag}_metrics.json")

            print("\n" + "=" * 80)
            print(f"[Chunk {i}/{len(unique_samples)}] sample={sample_value}")
            print(f"[Chunk {i}/{len(unique_samples)}] n_cells={n_cells}")
            print(f"[Chunk {i}/{len(unique_samples)}] out_h5ad={out_h5ad}")
            print(f"[Chunk {i}/{len(unique_samples)}] out_csv={out_csv}")

            if args.skip_done and os.path.exists(out_h5ad) and os.path.exists(out_csv):
                print("[Skip] output h5ad and csv already exist.")
                summary_rows.append({
                    "chunk_id": i,
                    "sample": str(sample_value),
                    "n_cells": n_cells,
                    "output_h5ad": out_h5ad,
                    "output_csv": out_csv,
                    "status": "skipped_exists",
                })
                continue

            try:
                t_chunk0 = time.perf_counter()
                sub = adata_b[cell_pos, :].to_memory()
                t_chunk1 = time.perf_counter()
                print(f"[Timing] load chunk into memory: {t_chunk1 - t_chunk0:.2f}s")

                sc_dataset = unicell_predict(
                    adata=sub,
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

                t_chunk2 = time.perf_counter()
                print(f"[Timing] inference: {t_chunk2 - t_chunk1:.2f}s")

                if args.compression == "gzip":
                    sc_dataset.adata.write_h5ad(
                        out_h5ad,
                        compression="gzip",
                        compression_opts=args.compression_opts
                    )
                else:
                    sc_dataset.adata.write_h5ad(out_h5ad)

                _save_prediction_csv(
                    adata=sc_dataset.adata,
                    sample_key=args.sample_key,
                    sample_value=str(sample_value),
                    out_csv=out_csv,
                )

                t_chunk3 = time.perf_counter()
                print(f"[Timing] write_h5ad+csv: {t_chunk3 - t_chunk2:.2f}s")
                print(f"[Done] saved -> {out_h5ad}")
                print(f"[Done] saved -> {out_csv}")

                row = {
                    "chunk_id": i,
                    "sample": str(sample_value),
                    "n_cells": n_cells,
                    "output_h5ad": out_h5ad,
                    "output_csv": out_csv,
                    "status": "ok",
                    "load_sec": round(t_chunk1 - t_chunk0, 4),
                    "infer_sec": round(t_chunk2 - t_chunk1, 4),
                    "write_sec": round(t_chunk3 - t_chunk2, 4),
                }

                if args.save_metrics_json:
                    metrics = _compute_metrics_for_adata(
                        sc_dataset.adata,
                        cell_type_key=args.cell_type_key,
                        tissue_key=args.tissue_key,
                        species_key=args.species_key,
                    )
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2, ensure_ascii=False)
                    print(f"[Done] metrics -> {out_json}")
                    row["metrics_json"] = out_json

                    if metrics.get("cell_type") is not None:
                        row["celltype_acc"] = metrics["cell_type"]["accuracy"]
                        row["celltype_macro_f1"] = metrics["cell_type"]["macro_f1"]
                    if metrics.get("tissue") is not None:
                        row["tissue_acc"] = metrics["tissue"]["accuracy"]
                        row["tissue_macro_f1"] = metrics["tissue"]["macro_f1"]
                    if metrics.get("species") is not None:
                        row["species_acc"] = metrics["species"]["accuracy"]
                        row["species_macro_f1"] = metrics["species"]["macro_f1"]

                summary_rows.append(row)

                del sub
                del sc_dataset
                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[ERROR] sample={sample_value} failed: {repr(e)}")
                summary_rows.append({
                    "chunk_id": i,
                    "sample": str(sample_value),
                    "n_cells": n_cells,
                    "output_h5ad": out_h5ad,
                    "output_csv": out_csv,
                    "status": f"failed: {repr(e)}",
                })
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

    finally:
        if adata_b is not None:
            try:
                adata_b.file.close()
            except Exception:
                pass

    t1 = time.perf_counter()
    print("\n" + "=" * 80)
    print(f"[All Done] total time: {t1 - t0:.2f}s")

    if args.summary_csv is None:
        summary_csv = os.path.join(args.out_dir, "chunk_summary.csv")
    else:
        summary_csv = args.summary_csv

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"[Summary] saved -> {summary_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
