#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :graph
# @Time      :2024/3/26 11:34
# @Author    :Luni Hu
import json

import scanpy as sc
from .ontoGRAPH import OntoGRAPH
import numpy as np
from networkx import read_gml
import os
from unicell.repo.scgpt.tokenizer.gene_tokenizer import GeneVocab

class scDataset(object):
    def __init__(self,
                 adata=None,
                 data_path=None,
                 cell_type_key=None,
                 batch_key=None,
                 trained=False,
                 graph=None,
                 highly_variable_genes=True,
                 llm_vocab=None,
                 llm_args=None):
        self.data_path = data_path
        self.trained = trained

        self.cell_type_key = cell_type_key
        self.adata = self.read_data(highly_variable_genes, True) if adata is None else adata
        if cell_type_key:
            cell_type = self.adata.obs[self.cell_type_key].unique().tolist()

        self.batch_key = batch_key
        if batch_key:
            self.adata.obs[self.batch_key] = self.adata.obs[self.batch_key].astype("category")
            self.batch_index = self.adata.obs[self.batch_key].cat.codes

        if trained:
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            if graph is None:
                graph = read_gml(os.path.join(curr_dir, "graph.gml"))
            self.ontograph = OntoGRAPH(graph=graph, cell_type=cell_type)
            self.adata.obs[cell_type_key] = [
                self.ontograph.obsolete_dict.get(c, c) for c in self.adata.obs[cell_type_key]]
            self.adata.obs["cell_type"] = [self.ontograph.id2name.get(c, "unknown") for c in self.adata.obs[cell_type_key]]
            self.adata = self.adata[~self.adata.obs[cell_type_key].isin([self.ontograph.common_ancestor]), :]
            self.adata = self.adata[self.adata.obs[cell_type_key].isin(self.ontograph.vocab.keys()), :]
            self.cell_type_index = self.get_cell_type_index()

        if llm_vocab:
            self.llm_vocab = GeneVocab.from_file(llm_vocab)

        if llm_args:
            with open(llm_args, 'r') as file:
                self.llm_args = json.load(file)


    def read_data(self, highly_variable_genes=True, subset=True):

        adata = sc.read_h5ad(self.data_path)

        if adata.X.min() >= 0:

            if adata.n_vars > 1000:
                sc.pp.filter_cells(adata, min_genes=200)

            # Normalization
            normalize_total = False
            log1p = False
            # Normalization
            if adata.X.max() > 25:
                log1p = True

                if adata.X.max() - np.int32(adata.X.max()) == np.int32(0):
                    normalize_total = 1e4

            if normalize_total:
                sc.pp.normalize_total(adata, target_sum=normalize_total)
                print("Normalizing Data!")

            if log1p:
                sc.pp.log1p(adata)
                print("Transforming Data to Log1P!")

        # subset highly variable genes
        if highly_variable_genes and self.trained:

            sc.pp.highly_variable_genes(adata, subset=subset)

        return adata

    def get_cell_type_index(self):

        if self.cell_type_key:

            cell_ontology_idx = [self.ontograph.vocab.get(key, -1)
                                 for key in self.adata.obs[self.cell_type_key].tolist()]

        else:
            raise ValueError("cell_type_key parameter must be provided!")

        return cell_ontology_idx

    def get_initial_embeddings(self, input_dim, adata_trained=None, emb_key=None):

        self.adata.raw = self.adata.copy()
        if emb_key is not None:
            return self.adata.obsm[emb_key]

        if self.trained:

            if input_dim is not None:

                sc.pp.scale(self.adata)

                sc.pp.pca(self.adata, n_comps=input_dim)

            return self.adata.obsm["X_pca"]

        else:

            self.adata = self.adata[:, self.adata.var_names.isin(adata_trained.var_names)].copy()

            sc.pp.scale(self.adata)

            adata_trained = adata_trained[:, self.adata.var_names].copy()

            PCs = adata_trained.varm["PCs"]

            emb = np.dot(self.adata.X, PCs)

            arr_dtype = adata_trained.obsm["X_pca"].dtype

            emb = emb.astype(arr_dtype)

            return emb

    def get_cell_type_hierarchy_matrix(self):

        hierarchical_array = self.ontograph.hierarchical_array

        cell_type_idx = list(set(self.cell_type_index))

        targets = [array[cell_type_idx, :] for array in hierarchical_array]

        # targets = [target for target in targets if not torch.all(target == 0)]

        return targets
