#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :evaluate.py
# @Time      :2024/4/7 12:00
# @Author    :Luni Hu

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(scDataset, label_key, prediction_key):

    cell_type_set = set(scDataset.adata.obs[label_key]) | set(scDataset.adata.obs[prediction_key])

    cell_type_dict = {cell_type:idx for idx, cell_type in enumerate(cell_type_set)}

    labels = [cell_type_dict[i] for i in scDataset.adata.obs[label_key]]

    predictions = [cell_type_dict[i] for i in scDataset.adata.obs[prediction_key]]

    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    micro_f1 = f1_score(labels, predictions, average='micro')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1,
      'micro_f1': micro_f1
    }
