#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 11:57
# @Author  : qiuping
# @File    : utils.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/5/20 11:57  create file. 
"""
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch.distributed as dist
import pickle
import os
import networkx as nx
import munch
import toml
import json

def get_reduced(tensor, device, dest_device):
    """
    将不同GPU上的变量或tensor集中在主GPU上，并得到均值
    """
    tensor = tensor.clone().detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
    tensor = tensor.to(device)
    torch.distributed.reduce(tensor, dst=dest_device)
    tensor_mean = tensor.item() / torch.distributed.get_world_size()
    return tensor_mean


def average_scalar(local_scalar,  device):
    local_scalar = torch.tensor([local_scalar]).to(device)
    dist.all_reduce(local_scalar, op=dist.ReduceOp.SUM)
    global_mean = local_scalar / torch.distributed.get_world_size()
    return global_mean


def cal_model_params(model: torch.nn.Module) -> int:
    """
    calculate model parameters
    """
    model_param_count = 0
    for param in model.parameters():
        model_param_count += param.numel()
    return model_param_count


def compute_metrics(labels, predictions):
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    micro_f1 = f1_score(labels, predictions, average='micro')
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1
    }


def train_test_split_adata(adata, test_size=0.2):
    cell_indices = adata.obs.index
    cell_indices = cell_indices[~cell_indices.duplicated(keep='first')]
    train_indices, test_indices = train_test_split(cell_indices, test_size=test_size)
    print(len(cell_indices), len(train_indices), len(test_indices))
    train_data = adata[train_indices]
    test_data = adata[test_indices]
    return train_data, test_data


def load_model_params(model_path, model):
    model_dict = torch.load(model_path)
    params = {}
    for k, v in model_dict["model_state_dict"].items():
        k = k.replace('module.', '')
        params[k] = v
    model.load_state_dict(params)
    return model


def load_ontograph(output):
    with open(os.path.join(output, 'ontoGraph.pk'), 'rb') as w:
        print(os.path.join(output, 'ontoGraph.pk'))
        onto_graph = pickle.load(w)
    g = nx.read_gml(os.path.join(output, 'ontoGraph.graph.gml'))
    onto_graph.graph = g
    return onto_graph

def load_config(config_file):
    args = munch.munchify(toml.load(config_file))
    if args.model_used in ('scgpt', 'scmamba'):
        with open(args.model_param_file, 'r') as fd:
            params = json.load(fd)
        for p in params:
            if p not in args:
                args[p] = params[p]
    return args