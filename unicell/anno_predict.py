#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/29 9:34
# @Author  : qiuping
# @File    : anno_predict.py.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/5/29 9:34  create file. 
"""
import os
import pickle
import sys
import numpy as np
import pandas as pd

cur_file = os.path.dirname(__file__)
sys.path.insert(0, cur_file + "/../")
import torch
from unicell.hmcn import HMCN
from unicell.utils.utils import load_ontograph
from unicell.scDataset import scDataset

from scipy import sparse
import anndata as ad
import networkx as nx
from unicell.dataset import HMCNDataset
from unicell.repo.geneformer.in_silico_perturber import get_model_input_size, pad_tensor_list
from torch.utils.data import DataLoader


def unicell_predict(adata,
                   filepath,
                   ckpt_dir=None,
                   batch_size=512,
                   device=None):
    model_path = os.path.join(ckpt_dir, 'unicell_v1.best.pth')
    model_dict = torch.load(model_path)
    metadata = model_dict["metadata"]

    sc_dataset = read_data(adata,
                           filepath,
                           ckpt_dir,
                           metadata["llm_vocab_file"],
                           metadata["llm_args_file"])
    hierarchical_array = sc_dataset.ontograph.hierarchical_array
    num_classes = len(hierarchical_array)
    print("num_class:", num_classes)
    hierarchical_class = [arr.shape[1] for arr in hierarchical_array]
    hierarchical_depth = [metadata["global_layer"] if i > 0 else 0 for i in range(num_classes)]
    global2local = [metadata["local_layer"] if i > 0 else 0 for i in range(num_classes)]
    with open(os.path.join(ckpt_dir, 'celltype_dict.pk'), 'rb') as f:
        label_dict = pickle.load(f)
    cls2id = {v:sc_dataset.ontograph.vocab[k] for k, v in label_dict.items()}

    model = HMCN(input_type=metadata["input_type"],
                 input_dim=metadata["input_dim"],
                 output_dim=metadata["output_dim"],
                 num_classes=num_classes,
                 hierarchical_depth=hierarchical_depth,
                 global2local=global2local,
                 hierarchical_class=hierarchical_class,
                 hidden_layer_dropout=metadata["hidden_layer_dropout"],
                 cls_num=len(label_dict),
                 llm_model_file=metadata["llm_model_file"],
                 llm_vocab_file=metadata["llm_vocab_file"],
                 llm_args_file=metadata["llm_args_file"])

    params = {}
    for k, v in model_dict["model_state_dict"].items():
        k = k.replace('module.', '')
        params[k] = v
    model.load_state_dict(params)
    model = model.eval().to(device)

    global_outs = []
    local_outs = []
    cls_outs = []
    cell_embs = []

    sc_dataset.cell_type_index = [0]*len(sc_dataset.adata)
    sc_dataset.cell_type_key = "cell_type_pseudo"
    sc_dataset.adata.obs[sc_dataset.cell_type_key] = [0]*len(sc_dataset.adata)

    dataset = HMCNDataset(sc_dataset, input_type=metadata["input_type"])
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=collate_fn_with_args(input_type=metadata["input_type"], model=model))
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for _, (batch_data, batch_labels, cls_labels) in enumerate(dataloader):

            if metadata["input_type"] in "GeneFormer":
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                model.geneformer = model.geneformer.to(device)
            elif metadata["input_type"] == "scGPT":
                batch_data = {
                    "input_ids": torch.stack([b["input_ids"] for b in batch_data]).to(torch.long).to(device),
                    "values": torch.stack([b["values"] for b in batch_data]).to(device)}
            else:
                batch_data = torch.stack(batch_data).to(device)

            with torch.cuda.amp.autocast(enabled=(metadata["input_type"] == "scGPT")):
                encoder_x, global_out, local_out, cls_out = model(batch_data)

            global_outs.append(global_out)
            local_outs.append(local_out)
            cls_outs.append(cls_out)
            cell_embs.append(encoder_x.detach().cpu().numpy())

    cls_layer_output = torch.cat(cls_outs, dim=0).detach().cpu().numpy()
    labels_pred = [cls2id[idx] for idx in np.argmax(cls_layer_output, axis=1)]
    vocab = {v:k for k, v in sc_dataset.ontograph.vocab.items()}
    sc_dataset.adata.obs["predicted_cell_type_ontology_id"] = [vocab[i] for i in labels_pred]
    sc_dataset.adata.obsm['unicell_emb'] = np.concatenate(cell_embs, axis=0)
    sc_dataset.adata.obsm['cls_emb'] = cls_layer_output
    sc_dataset.adata.uns["cls_cell_type"] = [sc_dataset.ontograph.id2name[vocab[cls2id[id]]] for id in range(len(cls2id))]
    sc_dataset.adata.obs["predicted_cell_type"] = [sc_dataset.ontograph.id2name[i] for i in
                                                   sc_dataset.adata.obs["predicted_cell_type_ontology_id"]]
    celltype_preds = list(set(sc_dataset.adata.obs["predicted_cell_type_ontology_id"]))
    hierarchical_level = ["level_"+str(i) for i in range(max(sc_dataset.ontograph.hierarchical_levels.values())+1)]
    hierarchical_level_df = pd.DataFrame(index=celltype_preds, columns=hierarchical_level)
    lca = sc_dataset.ontograph.common_ancestor
    for i, celltype in enumerate(celltype_preds):
        path = nx.shortest_path(sc_dataset.ontograph.graph, lca, celltype)
        hierarchical_level_df.iloc[i, :len(path)] = [sc_dataset.ontograph.id2name[node] for node in path]
    hierarchical_level_df = hierarchical_level_df.loc[sc_dataset.adata.obs["predicted_cell_type_ontology_id"]]
    hierarchical_level_df.index = sc_dataset.adata.obs_names
    sc_dataset.adata.obs = pd.concat([sc_dataset.adata.obs, hierarchical_level_df], axis = 1)

    return sc_dataset


def read_data(adata, filepath, ckpt_dir, llm_vocab, llm_args):
    scdataset = scDataset(adata=adata,
                          data_path=filepath,
                          cell_type_key=None,
                          trained=False,
                          highly_variable_genes=False,
                          llm_vocab=llm_vocab,
                          llm_args=llm_args)
    gene_path = os.path.join(ckpt_dir, 'gene_names.pk')
    with open(gene_path, 'rb') as f:
        gene_names = pickle.load(f)

    scdataset.ontograph = load_ontograph(ckpt_dir)

    new_data = np.zeros((scdataset.adata.X.shape[0], len(gene_names)))
    useful_gene_index = np.where(scdataset.adata.var_names.isin(gene_names))
    useful_gene = scdataset.adata.var_names[useful_gene_index]
    if len(useful_gene) == 0:
        raise ValueError("No gene names in ref gene, please check the adata.var_names are gene Symbol!")
    print('useful gene index: {}'.format(len(useful_gene)))
    gene_names = list(gene_names)
    gene_index = [gene_names.index(i) for i in useful_gene]
    if not sparse.issparse(scdataset.adata.X):
        new_data[:, gene_index] = scdataset.adata[:, useful_gene_index[0]].X
    else:
        print(scdataset.adata[:, useful_gene_index[0]])
        new_data[:, gene_index] = scdataset.adata[:, useful_gene_index[0]].X.toarray()
    new_data = sparse.csr_matrix(new_data)
    new_adata = ad.AnnData(X=new_data,
                           obs=scdataset.adata.obs,
                           obsm=scdataset.adata.obsm,
                           uns=scdataset.adata.uns
                           )
    new_adata.var_names = gene_names
    scdataset.adata = new_adata

    return scdataset


def collate_fn_with_args(input_type, model):
    def collate_fn(batch):
        batch_data, batch_labels, cls_labels = zip(*batch)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)  # Ensure correct type for batch_labels

        # If the input type is GeneFormer
        if input_type == "GeneFormer":

            model_input_size = get_model_input_size(model.geneformer)

            # Find the maximum sequence length in the batch
            max_len = max(data["length"] for data in batch_data)

            # Extract and pad `input_ids`
            input_data_minibatch = [torch.tensor(data["input_ids"], dtype=torch.long) for data in batch_data]
            pad_token_id = model.llm_vocab["<pad>"]
            input_data_minibatch = pad_tensor_list(
                input_data_minibatch, max_len, pad_token_id, model_input_size
            )

            # Prepare `new_batch_data` with padded inputs and lengths
            new_batch_data = {
                "input_ids": input_data_minibatch,  # Stack padded input tensors
                "length": torch.tensor([data["length"] for data in batch_data], dtype=torch.long)
            }
            return new_batch_data, batch_labels, cls_labels

        # If input type is not GeneFormer, return data as is
        return batch_data, batch_labels, cls_labels

    return collate_fn
