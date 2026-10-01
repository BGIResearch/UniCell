#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

import numpy as np
import pandas as pd

# match your UCE style (TLS/OpenMP)
import sklearn
from sklearn.utils import _openmp_helpers as _sk_omp
print("sklearn OpenMP enabled:", _sk_omp._openmp_parallelism_enabled())

import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
from unicell.anno_predict import unicell_predict


# -------------------------
# UMAP utils (same as UCE)
# -------------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def make_clean(adata_in: ad.AnnData, emb_key: str):
    n = adata_in.n_obs
    X_empty = np.zeros((n, 0), dtype=np.float32)
    obs = adata_in.obs.copy()
    var = pd.DataFrame(index=pd.Index([], name="var_names"))
    X_emb = adata_in.obsm[emb_key]
    if not isinstance(X_emb, np.ndarray):
        X_emb = np.asarray(X_emb)
    return ad.AnnData(X=X_empty, obs=obs, var=var, obsm={emb_key: X_emb.copy()})


def make_palette(n: int):
    phi = 0.618033988749895
    hues = (phi * np.arange(n)) % 1.0
    cols = [mcolors.to_hex(mcolors.hsv_to_rgb((h, 0.85, 0.95))) for h in hues]
    return cols


def compute_umap_once(a: ad.AnnData, rep_key: str, n_neighbors: int, min_dist: float, random_state: int, neighbors_key: str):
    sc.pp.neighbors(
        a,
        use_rep=rep_key,
        n_neighbors=n_neighbors,
        random_state=random_state,
        key_added=neighbors_key,
    )
    sc.tl.umap(
        a,
        min_dist=min_dist,
        random_state=random_state,
        neighbors_key=neighbors_key,
    )


def draw_umap(a: ad.AnnData, key: str, figdir: str, size: float, alpha: float, show_legend: bool, prefix: str):
    if key not in a.obs.columns:
        print(f"[WARN] {key} not in obs. Skip.")
        return
    a.obs[key] = a.obs[key].astype(str).astype("category")
    n_cat = a.obs[key].cat.categories.size
    palette = make_palette(n_cat)

    out_png = os.path.join(figdir, f"{prefix}_by_{key}.png")
    fig = sc.pl.umap(
        a,
        color=key,
        palette=palette,
        show=False,
        return_fig=True,
        size=size,
        alpha=alpha,
        legend_loc="right margin" if show_legend else None,
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_png} (legend={'ON' if show_legend else 'OFF'})")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("UniCell UMAP (shared + heads)")

    ap.add_argument("--test_h5ad", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", default="./unicell_umap")

    # color keys (GT by default; can switch to predicted_* easily)
    ap.add_argument("--species_col", default="organism")
    ap.add_argument("--tissue_col", default="general_tissue")
    ap.add_argument("--celltype_col", default="cell_type_ontology_term_id")  # GT ontology id

    # embeddings produced by anno_predict.py
    ap.add_argument("--shared_emb_key", default="unicell_emb")
    ap.add_argument("--species_head_key", default="species_cls_emb")
    ap.add_argument("--tissue_head_key", default="tissue_cls_emb")
    ap.add_argument("--celltype_head_key", default="cls_emb")

    # inference
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--no_metrics", action="store_true", help="Disable metrics inside unicell_predict")

    # umap params
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.5)
    ap.add_argument("--random-state", type=int, default=0)

    # plot params
    ap.add_argument("--legend", action="store_true")
    ap.add_argument("--pt-size", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.8)

    # save
    ap.add_argument("--no_write_h5ad", action="store_true")

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    out_shared = os.path.join(args.out_dir, "umap_shared")
    out_heads = os.path.join(args.out_dir, "umap_heads")
    ensure_dir(out_shared)
    ensure_dir(out_heads)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("Device:", device)

    print("[1/4] Load test h5ad ...")
    adata = sc.read_h5ad(args.test_h5ad)

    print("[2/4] UniCell inference (write obs/obsm/uns) ...")
    sc_dataset = unicell_predict(
        adata=adata,
        filepath=args.test_h5ad,
        ckpt_dir=args.ckpt_dir,
        batch_size=args.batch_size,
        device=device,
        compute_metrics=(not args.no_metrics),
        cell_type_key=args.celltype_col,
        tissue_key=args.tissue_col,
        species_key=args.species_col,
    )
    adata = sc_dataset.adata

    # check obsm keys
    for k in [args.shared_emb_key, args.species_head_key, args.tissue_head_key, args.celltype_head_key]:
        if k not in adata.obsm:
            raise KeyError(f"Missing obsm['{k}']. Available: {list(adata.obsm.keys())}")

    print("[3/4] Plot UMAPs ...")

    # A) shared: compute once, recolor 3 keys
    a_shared = make_clean(adata, args.shared_emb_key)
    print(f"Computing UMAP for {args.shared_emb_key} ...")
    compute_umap_once(
        a_shared,
        rep_key=args.shared_emb_key,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        neighbors_key="neighbors_shared",
    )
    draw_umap(a_shared, args.species_col, out_shared, args.pt_size, args.alpha, args.legend, prefix=f"{args.shared_emb_key}_umap")
    draw_umap(a_shared, args.tissue_col, out_shared, args.pt_size, args.alpha, args.legend, prefix=f"{args.shared_emb_key}_umap")
    draw_umap(a_shared, args.celltype_col, out_shared, args.pt_size, args.alpha, args.legend, prefix=f"{args.shared_emb_key}_umap")

    # B) heads: each compute once, draw its own key
    a_sp = make_clean(adata, args.species_head_key)
    compute_umap_once(a_sp, args.species_head_key, args.n_neighbors, args.min_dist, args.random_state, "neighbors_species_head")
    draw_umap(a_sp, args.species_col, out_heads, args.pt_size, args.alpha, args.legend, prefix=f"{args.species_head_key}_umap")

    a_ti = make_clean(adata, args.tissue_head_key)
    compute_umap_once(a_ti, args.tissue_head_key, args.n_neighbors, args.min_dist, args.random_state, "neighbors_tissue_head")
    draw_umap(a_ti, args.tissue_col, out_heads, args.pt_size, args.alpha, args.legend, prefix=f"{args.tissue_head_key}_umap")

    a_ct = make_clean(adata, args.celltype_head_key)
    compute_umap_once(a_ct, args.celltype_head_key, args.n_neighbors, args.min_dist, args.random_state, "neighbors_celltype_head")
    draw_umap(a_ct, args.celltype_col, out_heads, args.pt_size, args.alpha, args.legend, prefix=f"{args.celltype_head_key}_umap")

    print("[4/4] Save h5ad/meta ...")
    out_h5ad = os.path.join(args.out_dir, "test_with_unicell_features.h5ad")
    if not args.no_write_h5ad:
        adata.write_h5ad(out_h5ad)
        print("Saved:", out_h5ad)

    meta = {
        "input_h5ad": args.test_h5ad,
        "ckpt_dir": args.ckpt_dir,
        "device": device,
        "saved_h5ad": None if args.no_write_h5ad else out_h5ad,
        "obsm_used": {
            "shared": args.shared_emb_key,
            "species_head": args.species_head_key,
            "tissue_head": args.tissue_head_key,
            "celltype_head": args.celltype_head_key,
        },
        "color_keys": {"species": args.species_col, "tissue": args.tissue_col, "celltype": args.celltype_col},
        "umap_params": {"n_neighbors": args.n_neighbors, "min_dist": args.min_dist, "random_state": args.random_state},
        "plot_params": {"pt_size": args.pt_size, "alpha": args.alpha, "legend": bool(args.legend)},
    }
    with open(os.path.join(args.out_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
