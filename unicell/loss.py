#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :loss.py
# @Time      :2024/4/12 18:00
# @Author    :Luni Hu

import torch
import torch.nn as nn
import torch.nn.functional as F

class HMCNLoss(nn.Module):
    def __init__(self, input_type, scDataset):

        super(HMCNLoss, self).__init__()
        self.hierarchical_array = scDataset.ontograph.hierarchical_array
        self.hierarchical_level = scDataset.ontograph.hierarchical_levels
        self.num_levels = len(scDataset.ontograph.hierarchical_array)
        self.idx2id = {v:k for k, v in scDataset.ontograph.vocab.items()}
        self.input_type = input_type

    def forward(self, global_layer_output, local_layer_outputs, batch_labels):

        levels = [self.hierarchical_level[self.idx2id[label.item()]] for label in batch_labels]

        if self.input_type == 'scGPT':

            batch_global_labels = torch.tensor(levels, device=global_layer_output.device)
            global_loss = F.cross_entropy(global_layer_output, batch_global_labels)

            targets = [self.hierarchical_array[idx][batch_labels, :] for idx in range(self.num_levels)]

            local_loss = 0

            for level in range(self.num_levels - 1):
                output = local_layer_outputs[level]
                target_level = torch.argmax(targets[level + 1], dim=1).to(output.device)
                output = output.to(target_level.device)
                local_loss += F.cross_entropy(output, target_level)

        else:

            batch_global_labels = torch.zeros(len(levels), self.num_levels)

            for idx, level in enumerate(levels):
                batch_global_labels[idx, :(level + 1)] = 1

            batch_global_labels = batch_global_labels.to(global_layer_output.dtype).to(global_layer_output.device)
            global_loss = F.binary_cross_entropy(global_layer_output, batch_global_labels)
            targets = [self.hierarchical_array[idx][batch_labels, :] for idx in range(self.num_levels)]

            local_loss = 0

            for level in range(self.num_levels - 1):
                output = local_layer_outputs[level]
                target_level = targets[level + 1]
                target_level = target_level.to(output.dtype).to(output.device)
                local_loss += F.binary_cross_entropy(output, target_level)
        return global_loss, local_loss


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):

        effective_target = torch.eye(self.num_classes, device=targets.device)[targets]

        # Calculate Cross entropy
        logit = F.softmax(inputs, dim=1)
        logit = logit.clamp(1e-7, 1.0)
        effective_target = effective_target.to(logit.dtype)
        ce = -(effective_target * torch.log(logit))

        # Calculate Focal Loss
        weight = torch.pow(-logit + 1., self.gamma)
        fl = ce * weight * self.alpha

        if self.reduction == 'sum':
            return fl.sum()
        elif self.reduction == 'mean':
            return fl.mean()

def compute_mmd(features, batch_labels):
    # Compute the maximum mean discrepancy (MMD) loss between different batches
    num_batches = batch_labels.unique().size(0)
    mmd_loss = 0
    if num_batches > 1:
        for i in range(num_batches):
            for j in range(i+1, num_batches):
                batch_i_mask = batch_labels == i
                batch_j_mask = batch_labels == j
                batch_i_features = features[batch_i_mask]
                batch_j_features = features[batch_j_mask]
                mmd_loss += torch.norm(batch_i_features.mean(dim=0) - batch_j_features.mean(dim=0))
        return mmd_loss / (num_batches * (num_batches - 1) // 2)
    else:
        return mmd_loss


