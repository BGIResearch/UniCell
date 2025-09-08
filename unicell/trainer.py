#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/8 15:17
# @Author  : Luni Hu
# @File    : trainer.py
# @Software: PyCharm

import os
import numpy as np
import copy
import pickle
import torch
import torch.optim as optim
from unicell.loss import HMCNLoss, FocalLoss
from unicell.dataset import HMCNDataset
from unicell.hmcn import HMCN
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import time
from unicell.utils.utils import compute_metrics as compute_metrics_cls
from unicell.repo.geneformer.in_silico_perturber import get_model_input_size, pad_tensor_list

class UnicellTrainer:
    def __init__(self,
                 scDataset,
                 input_type,
                 input_dim,
                 output_dim,
                 batch_size,
                 learning_rate,
                 num_epochs,
                 beta,
                 device,
                 global_layer,
                 local_layer,
                 hidden_layer_dropout,
                 ckpt_dir,
                 ddp_train=False,
                 save_epoch=False,
                 local_rank=0,
                 llm_model_file=None,
                 llm_vocab_file=None,
                 llm_args_file=None):
        self.local_rank = local_rank
        self.ddp_train = ddp_train
        self.ckpt_dir = ckpt_dir
        self.input_type = input_type
        self.input_dim = input_dim
        self.output_dim = output_dim

        hierarchical_array = scDataset.get_cell_type_hierarchy_matrix()
        self.num_classes = len(hierarchical_array)
        print("num_classes", self.num_classes)
        hierarchical_class = [arr.shape[1] for arr in hierarchical_array]
        print(hierarchical_class)
        self.global_layer = global_layer
        self.local_layer = local_layer
        hierarchical_depth = [self.global_layer if i > 0 else 0 for i in range(self.num_classes)]
        global2local = [self.local_layer if i > 0 else 0 for i in range(self.num_classes)]
        labels = scDataset.adata.obs[scDataset.cell_type_key].unique().tolist()
        self.label_dict = {label: i for i, label in enumerate(labels)}
        with open(os.path.join(self.ckpt_dir, 'celltype_dict.pk'), 'wb') as w:
            pickle.dump(self.label_dict, w)
        self.cls2id = {v:scDataset.ontograph.vocab[k] for k, v in self.label_dict.items()}
        self.hidden_layer_dropout = hidden_layer_dropout

        self.llm_model_file = llm_model_file
        self.llm_vocab_file = llm_vocab_file
        self.llm_args_file = llm_args_file

        model = HMCN(self.input_type,
                     self.input_dim,
                     self.output_dim,
                     self.num_classes,
                     hierarchical_depth,
                     global2local,
                     hierarchical_class,
                     self.hidden_layer_dropout,
                     len(labels),
                     self.llm_model_file,
                     self.llm_vocab_file,
                     self.llm_args_file)
        self.model = model.to(device)

        dataset = HMCNDataset(scDataset, self.input_type)
        sampler = RandomSampler(dataset) if not ddp_train else DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True, collate_fn=self.collate_fn)
        self.dataloader = dataloader

        if self.ddp_train:
            self.model = DistributedDataParallel(self.model, device_ids=[local_rank],
                                                 output_device=local_rank)

        self.criterion = HMCNLoss(input_type=self.input_type, scDataset=scDataset)
        self.learning_rate = learning_rate

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4 if self.input_type == 'scGPT' else 1e-8)
        self.num_epochs = num_epochs
        self.beta = beta
        self.device = device
        self.cell_type_idx_constrained = list(set(scDataset.cell_type_index))
        self.cell_type_idx_set = scDataset.ontograph.cell_type_idx_set
        self.best_f1 = 0
        self.best_loss = float("Inf")
        self.best_model = None
        self.batch_size = batch_size
        self.best_epoch = 0
        self.cls_criterion = FocalLoss(num_classes=len(labels))

        self.save_epoch = save_epoch

        # Print the initial settings
        self.print_initial_settings()

    def print_initial_settings(self):
        print("=== Initial Settings ===")
        print(f"Local Rank: {self.local_rank}")
        print(f"DDP Training: {self.ddp_train}")
        print(f"Checkpoint Directory: {self.ckpt_dir}")
        print(f"Input Type: {self.input_type}")
        print(f"Input Dimensions: {self.input_dim}")
        print(f"Output Dimensions: {self.output_dim}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Number of Epochs: {self.num_epochs}")
        print(f"Beta: {self.beta}")
        print(f"Device: {self.device}")
        print(f"Global Layer: {self.global_layer}")
        print(f"Local Layer: {self.local_layer}")
        print(f"Hidden Layer Dropout: {self.hidden_layer_dropout}")
        print(f"LLM Model File: {self.llm_model_file}")
        print(f"LLM Vocab File: {self.llm_vocab_file}")
        print(f"LLM Args File: {self.llm_args_file}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Label Dictionary: {self.label_dict}")
        print(f"Optimizer Params: {self.optimizer}")
        print("=========================")

    def train(self, scdata_test=None):

        for epoch in range(self.num_epochs):
            self.model.train()

            start_time = time.time()
            epoch_loss = self.train_one_epoch(epoch)

            if scdata_test is not None:
                cls_eval_res = self.predict(scdata_test, batch_size=self.batch_size)
                acc = float(cls_eval_res['accuracy'])
                f1 = float(cls_eval_res['macro_f1'])
                end_time = time.time()
                execution_time = end_time - start_time
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}]/{execution_time:.2f}s,  Train Loss: {epoch_loss:.6f}, eval_Acc: {acc:.4f}, macro_f1: {f1:.4f}."
                    )

                if f1 >= self.best_f1:
                    self.best_f1 = f1
                    self.best_loss = epoch_loss
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = epoch

            else:
                end_time = time.time()
                execution_time = end_time - start_time
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}]/{execution_time:.2f}s,  Train Loss: {epoch_loss:.6f}."
                )

                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = epoch

        if self.local_rank == 0:
            metadata = {
                'input_type': self.input_type,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'global_layer': self.global_layer,
                'local_layer': self.local_layer,
                'hidden_layer_dropout': self.hidden_layer_dropout,
                'llm_model_file': self.llm_model_file,
                'llm_vocab_file': self.llm_vocab_file,
                'llm_args_file': self.llm_args_file
            }

            checkpoint = {
                'model_state_dict': self.best_model.state_dict(),
                'metadata': metadata
            }
            torch.save(checkpoint, os.path.join(self.ckpt_dir, 'unicell_v1.best.pth'))

    def train_one_epoch(self, epoch):
        if self.ddp_train:
            self.dataloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        self.optimizer.zero_grad()

        for batch_idx, (batch_data, batch_labels, cls_labels) in enumerate(self.dataloader):

            if self.input_type == "GeneFormer":
                batch_data = {k:v.to(self.device) for k, v in batch_data.items()}
                self.model.geneformer = self.model.geneformer.to(self.device)
            elif self.input_type == "scGPT":
                batch_data = {"input_ids": torch.stack([b["input_ids"] for b in batch_data]).to(torch.long).to(self.device),
                              "values": torch.stack([b["values"] for b in batch_data]).to(self.device)}
            else:
                batch_data = torch.stack(batch_data).to(self.device)
            self.optimizer.zero_grad()

            if self.input_type == "scGPT":

                with torch.cuda.amp.autocast(enabled=True):
                    _, global_layer_output, local_layer_outputs, global_cls_output = self.model(batch_data)
                    # global_layer_output = torch.sigmoid(global_layer_output)
                    # local_layer_outputs = [torch.sigmoid(local_layer_output) for local_layer_output in local_layer_outputs]
            else:
                _, global_layer_output, local_layer_outputs, global_cls_output = self.model(batch_data)

            # Compute loss
            with torch.cuda.amp.autocast(enabled=False):
                global_loss, local_loss = self.criterion(global_layer_output, local_layer_outputs, batch_labels)
            batch_cls_labels = [self.label_dict.get(label, -1) for label in cls_labels]
            batch_cls_labels = torch.tensor(batch_cls_labels, device=self.device)
            global_cls_loss = self.cls_criterion(global_cls_output, batch_cls_labels)
            # Calculate the total loss
            loss = global_loss * self.beta + local_loss + global_cls_loss

            # Backward pass
            loss.backward()
            # Update the parameters
            self.optimizer.step()
            running_loss += loss.item()
        # Calculate the average loss for the epoch
        epoch_loss = running_loss / batch_idx
        if self.save_epoch:
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, f'unicell_v1.ep{epoch}.pth'))
        return epoch_loss

    def predict(self, scDataset, batch_size):
        dataset = HMCNDataset(scDataset, input_type=self.input_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.collate_fn)
        cls_outs = []
        labels = []
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch_idx, (batch_data, batch_labels, cls_labels) in enumerate(dataloader):

                if self.input_type in "GeneFormer":
                    batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                    self.model.geneformer = self.model.geneformer.to(self.device)
                elif self.input_type == "scGPT":
                    batch_data = {
                        "input_ids": torch.stack([b["input_ids"] for b in batch_data]).to(torch.long).to(self.device),
                        "values": torch.stack([b["values"] for b in batch_data]).to(self.device)}
                else:
                    batch_data = torch.stack(batch_data).to(self.device)
                labels.extend(batch_labels.detach().cpu().numpy())

                with torch.cuda.amp.autocast(enabled=(self.input_type == "scGPT")):
                    _, _, _, cls_layer_outs = self.model(batch_data)
                cls_outs.append(cls_layer_outs)

        cls_layer_output = torch.cat(cls_outs, dim=0).detach().cpu().numpy()
        labels_pred = [self.cls2id[idx] for idx in np.argmax(cls_layer_output, axis =1)]
        cls_eval_res = compute_metrics_cls(labels, labels_pred)
        return cls_eval_res

    def collate_fn(self, batch):
        batch_data, batch_labels, cls_labels = zip(*batch)

        batch_labels = torch.tensor(batch_labels)

        if self.input_type == "GeneFormer":
            # Retrieve model input size
            model_input_size = get_model_input_size(self.model.geneformer)

            # Find the maximum sequence length in the batch
            max_len = max(data["length"] for data in batch_data)

            # Extract and pad `input_ids`
            input_data_minibatch = [torch.tensor(data["input_ids"], dtype=torch.long) for data in batch_data]
            pad_token_id = self.model.llm_vocab["<pad>"]
            input_data_minibatch = pad_tensor_list(
                input_data_minibatch, max_len, pad_token_id, model_input_size
            )

            # Prepare `new_batch_data` with padded inputs and lengths
            new_batch_data = {
                "input_ids": input_data_minibatch,  # Stack padded input tensors
                "length": torch.tensor([data["length"] for data in batch_data], dtype=torch.long)
            }

            return new_batch_data, batch_labels, cls_labels

        else:
            return batch_data, batch_labels, cls_labels