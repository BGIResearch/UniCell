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
import json
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
                 llm_args_file=None,
                 checkpoint_metadata_overrides=None):
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

        self.cls2id = {v: scDataset.ontograph.vocab[k] for k, v in self.label_dict.items()}

        self.tissue_key = getattr(scDataset, "tissue_key", None)
        self.species_key = getattr(scDataset, "species_key", None)

        self.tissue_label_dict = getattr(scDataset, "tissue_label_dict", None)
        self.species_label_dict = getattr(scDataset, "species_label_dict", None)

        tissue_cls_num = len(self.tissue_label_dict) if (self.tissue_label_dict is not None) else 0
        species_cls_num = len(self.species_label_dict) if (self.species_label_dict is not None) else 0

        if tissue_cls_num == 0 or species_cls_num == 0:
            raise ValueError(
                "tissue/species classes not found. "
                "Please construct scDataset(..., tissue_key='general_tissue', species_key='organism') "
                "and ensure these obs columns exist."
            )

        with open(os.path.join(self.ckpt_dir, 'tissue_dict.pk'), 'wb') as w:
            pickle.dump(self.tissue_label_dict, w)
        with open(os.path.join(self.ckpt_dir, 'species_dict.pk'), 'wb') as w:
            pickle.dump(self.species_label_dict, w)

        self.hidden_layer_dropout = hidden_layer_dropout
        self.llm_model_file = llm_model_file
        self.llm_vocab_file = llm_vocab_file
        self.llm_args_file = llm_args_file
        self.checkpoint_metadata_overrides = dict(checkpoint_metadata_overrides or {})

        model = HMCN(self.input_type,
                     self.input_dim,
                     self.output_dim,
                     self.num_classes,
                     hierarchical_depth,
                     global2local,
                     hierarchical_class,
                     self.hidden_layer_dropout,
                     len(labels),
                     tissue_cls_num,
                     species_cls_num,
                     self.llm_model_file,
                     self.llm_vocab_file,
                     self.llm_args_file)
        

        # ===== Parameter statistics (added here)=====
        def count_parameters(model):
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total, trainable

        total_params, trainable_params = count_parameters(model)

        print(f"[Model] Total params: {total_params}")
        print(f"[Model] Trainable params: {trainable_params}")
        print(f"[Model] Total (M): {total_params / 1e6:.2f} M")
        self.model = model.to(device)

        # dataloader
        self.dataset = HMCNDataset(scDataset, self.input_type)
        sampler = RandomSampler(self.dataset) if not ddp_train else DistributedSampler(self.dataset)
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=self.collate_fn
        )
        self.dataloader = dataloader

        if self.ddp_train:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )

        self.criterion = HMCNLoss(input_type=self.input_type, scDataset=scDataset)
        self.cls_criterion = FocalLoss(num_classes=len(labels))
        self.tissue_cls_criterion = FocalLoss(num_classes=tissue_cls_num)
        self.species_cls_criterion = FocalLoss(num_classes=species_cls_num)

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            eps=1e-4 if self.input_type == 'scGPT' else 1e-8
        )
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
        self.save_epoch = save_epoch

        self.print_initial_settings()

    def _base_model(self):
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

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
        print(f"Number of Classes (hier levels): {self.num_classes}")
        print(f"Celltype label dict size: {len(self.label_dict)}")
        print(f"Tissue key: {self.tissue_key}, classes: {len(self.tissue_label_dict) if self.tissue_label_dict else 0}")
        print(f"Species key: {self.species_key}, classes: {len(self.species_label_dict) if self.species_label_dict else 0}")
        print(f"Optimizer Params: {self.optimizer}")
        print("=========================")

    def train(self, scdata_test=None):
        if self.local_rank == 0:
            total_start = time.time()
            epoch_times = []

        for epoch in range(self.num_epochs):
            self.model.train()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            epoch_start = time.time()

            epoch_loss = self.train_one_epoch(epoch)

            if scdata_test is not None:
                cls_eval_res = self.predict(scdata_test, batch_size=self.batch_size)
                acc = float(cls_eval_res['accuracy'])
                f1 = float(cls_eval_res['macro_f1'])

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                epoch_end = time.time()
                epoch_time = epoch_end - epoch_start

                if self.local_rank == 0:
                    epoch_times.append(epoch_time)
                    print(
                        f"Epoch [{epoch + 1}/{self.num_epochs}]/{epoch_time:.2f}s,  "
                        f"Train Loss: {epoch_loss:.6f}, eval_Acc: {acc:.4f}, macro_f1: {f1:.4f}."
                    )

                if f1 >= self.best_f1:
                    self.best_f1 = f1
                    self.best_loss = epoch_loss
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = epoch

            else:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                epoch_end = time.time()
                epoch_time = epoch_end - epoch_start

                if self.local_rank == 0:
                    epoch_times.append(epoch_time)
                    print(
                        f"Epoch [{epoch + 1}/{self.num_epochs}]/{epoch_time:.2f}s,  "
                        f"Train Loss: {epoch_loss:.6f}."
                    )

                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = epoch

        if self.local_rank == 0:
            total_end = time.time()
            total_time = total_end - total_start
            avg_epoch_time = sum(epoch_times) / len(epoch_times) if len(epoch_times) > 0 else 0.0

            print("\n===== Training Time Summary =====")
            print(f"Total epochs      : {self.num_epochs}")
            print(f"Total train time  : {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
            print(f"Avg time / epoch  : {avg_epoch_time:.2f} seconds")
            print(f"Best epoch (by f1 or loss): {self.best_epoch + 1}")
            print("=================================\n")

            time_stats = {
                "total_epochs": self.num_epochs,
                "total_time_seconds": float(total_time),
                "total_time_minutes": float(total_time / 60.0),
                "avg_epoch_time_seconds": float(avg_epoch_time),
                "epoch_times_seconds": [float(t) for t in epoch_times],
                "best_epoch": int(self.best_epoch + 1),
            }
            time_stats_path = os.path.join(self.ckpt_dir, "train_time_summary.json")
            with open(time_stats_path, "w", encoding="utf-8") as f:
                json.dump(time_stats, f, indent=2, ensure_ascii=False)
            print(f"[TimeSummary] Saved training time stats to: {time_stats_path}")

            metadata = {
                'input_type': self.input_type,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'global_layer': self.global_layer,
                'local_layer': self.local_layer,
                'hidden_layer_dropout': self.hidden_layer_dropout,
                'llm_model_file': self.llm_model_file,
                'llm_vocab_file': self.llm_vocab_file,
                'llm_args_file': self.llm_args_file,
                'tissue_key': self.tissue_key,
                'species_key': self.species_key,
            }
            metadata.update(self.checkpoint_metadata_overrides)

            checkpoint = {
                'model_state_dict': self.best_model.state_dict(),
                'metadata': metadata
            }
            torch.save(checkpoint, os.path.join(self.ckpt_dir, 'unicell_v1.best.pth'))

    def train_one_epoch(self, epoch):
        if self.ddp_train:
            self.dataloader.sampler.set_epoch(epoch)

        running_loss = 0.0
        epoch_t0 = time.time()

        time_stats = {
            "dataloader_wait": 0.0,
            "unpack_batch": 0.0,
            "prepare_batch": 0.0,
            "zero_grad": 0.0,
            "forward": 0.0,
            "hier_loss": 0.0,
            "aux_loss": 0.0,
            "backward": 0.0,
            "optim_step": 0.0,
            "batch_total_inside": 0.0,
        }

        first_batch_stats = None
        prev_end_time = time.time()
        num_batches = len(self.dataloader)

        for batch_idx, batch in enumerate(self.dataloader):
            batch_enter_time = time.time()

            dataloader_wait_t = batch_enter_time - prev_end_time
            time_stats["dataloader_wait"] += dataloader_wait_t

            batch_inner_t0 = time.time()

            t0 = time.time()
            if len(batch) == 5:
                batch_data, batch_labels, cls_labels, tissue_labels, species_labels = batch
                batch_batch_labels = None
            else:
                batch_data, batch_labels, cls_labels, tissue_labels, species_labels, batch_batch_labels = batch
            unpack_t = time.time() - t0
            time_stats["unpack_batch"] += unpack_t

            t0 = time.time()
            if self.input_type == "GeneFormer":
                base_model = self._base_model()
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                base_model.geneformer = base_model.geneformer.to(self.device)
            elif self.input_type == "scGPT":
                batch_data = {
                    "input_ids": torch.stack([b["input_ids"] for b in batch_data]).to(torch.long).to(self.device),
                    "values": torch.stack([b["values"] for b in batch_data]).to(self.device)
                }
            elif self.input_type == "expr":
                batch_data = batch_data.to(self.device, non_blocking=True)
            else:
                batch_data = torch.stack(batch_data).to(self.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prepare_t = time.time() - t0
            time_stats["prepare_batch"] += prepare_t

            t0 = time.time()
            self.optimizer.zero_grad()
            zero_grad_t = time.time() - t0
            time_stats["zero_grad"] += zero_grad_t

            t0 = time.time()
            if self.input_type == "scGPT":
                with torch.cuda.amp.autocast(enabled=True):
                    (_, global_layer_output, local_layer_outputs,
                     global_cls_output, tissue_cls_output, species_cls_output) = self.model(batch_data)
            else:
                (_, global_layer_output, local_layer_outputs,
                 global_cls_output, tissue_cls_output, species_cls_output) = self.model(batch_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_t = time.time() - t0
            time_stats["forward"] += forward_t

            t0 = time.time()
            with torch.cuda.amp.autocast(enabled=False):
                global_loss, local_loss = self.criterion(global_layer_output, local_layer_outputs, batch_labels)
            hier_loss_t = time.time() - t0
            time_stats["hier_loss"] += hier_loss_t

            t0 = time.time()
            batch_cls_labels = torch.tensor(
                [self.label_dict.get(label, -1) for label in cls_labels],
                device=self.device
            )
            global_cls_loss = self.cls_criterion(global_cls_output, batch_cls_labels)

            tissue_labels = torch.tensor(tissue_labels, device=self.device, dtype=torch.long)
            species_labels = torch.tensor(species_labels, device=self.device, dtype=torch.long)

            tissue_cls_loss = self.tissue_cls_criterion(tissue_cls_output, tissue_labels)
            species_cls_loss = self.species_cls_criterion(species_cls_output, species_labels)

            loss = global_loss * self.beta + local_loss + global_cls_loss + tissue_cls_loss + species_cls_loss
            aux_loss_t = time.time() - t0
            time_stats["aux_loss"] += aux_loss_t

            t0 = time.time()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_t = time.time() - t0
            time_stats["backward"] += backward_t

            t0 = time.time()
            self.optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            optim_step_t = time.time() - t0
            time_stats["optim_step"] += optim_step_t

            running_loss += loss.item()

            batch_total_inside_t = time.time() - batch_inner_t0
            time_stats["batch_total_inside"] += batch_total_inside_t

            if batch_idx == 0:
                batch_shape = None
                batch_dtype = None
                batch_device = None

                if torch.is_tensor(batch_data):
                    batch_shape = tuple(batch_data.shape)
                    batch_dtype = str(batch_data.dtype)
                    batch_device = str(batch_data.device)
                elif isinstance(batch_data, dict):
                    batch_shape = {
                        k: tuple(v.shape) if torch.is_tensor(v) else type(v).__name__
                        for k, v in batch_data.items()
                    }
                    batch_dtype = {
                        k: str(v.dtype) if torch.is_tensor(v) else type(v).__name__
                        for k, v in batch_data.items()
                    }
                    batch_device = {
                        k: str(v.device) if torch.is_tensor(v) else "NA"
                        for k, v in batch_data.items()
                    }

                first_batch_stats = {
                    "batch_idx": batch_idx,
                    "dataloader_wait": dataloader_wait_t,
                    "unpack_batch": unpack_t,
                    "prepare_batch": prepare_t,
                    "zero_grad": zero_grad_t,
                    "forward": forward_t,
                    "hier_loss": hier_loss_t,
                    "aux_loss": aux_loss_t,
                    "backward": backward_t,
                    "optim_step": optim_step_t,
                    "batch_total_inside": batch_total_inside_t,
                    "batch_shape": batch_shape,
                    "batch_dtype": batch_dtype,
                    "batch_device": batch_device,
                    "loss": float(loss.item()),
                }

                if self.local_rank == 0:
                    print("\n[Batch 0 Detailed Timing]")
                    print(f"  dataloader_wait : {dataloader_wait_t:.2f}s")
                    print(f"  unpack_batch    : {unpack_t:.2f}s")
                    print(f"  prepare_batch   : {prepare_t:.2f}s")
                    print(f"  zero_grad       : {zero_grad_t:.2f}s")
                    print(f"  forward         : {forward_t:.2f}s")
                    print(f"  hier_loss       : {hier_loss_t:.2f}s")
                    print(f"  aux_loss        : {aux_loss_t:.2f}s")
                    print(f"  backward        : {backward_t:.2f}s")
                    print(f"  optim_step      : {optim_step_t:.2f}s")
                    print(f"  batch_total_in  : {batch_total_inside_t:.2f}s")
                    print(f"  batch_shape     : {batch_shape}")
                    print(f"  batch_dtype     : {batch_dtype}")
                    print(f"  batch_device    : {batch_device}")
                    print(f"  loss            : {loss.item():.6f}\n")

            prev_end_time = time.time()

        epoch_total_t = time.time() - epoch_t0
        epoch_loss = running_loss / len(self.dataloader)

        avg_stats = {k: (v / num_batches if num_batches > 0 else 0.0) for k, v in time_stats.items()}
        known_total = (
            time_stats["dataloader_wait"]
            + time_stats["unpack_batch"]
            + time_stats["prepare_batch"]
            + time_stats["zero_grad"]
            + time_stats["forward"]
            + time_stats["hier_loss"]
            + time_stats["aux_loss"]
            + time_stats["backward"]
            + time_stats["optim_step"]
        )
        other_time = max(epoch_total_t - known_total, 0.0)

        if self.local_rank == 0:
            print("\n[Epoch Timing Summary]")
            print(f"  epoch_total_time        : {epoch_total_t:.2f}s")
            print(f"  num_batches             : {num_batches}")
            print(f"  dataloader_wait_total   : {time_stats['dataloader_wait']:.2f}s")
            print(f"  unpack_batch_total      : {time_stats['unpack_batch']:.2f}s")
            print(f"  prepare_batch_total     : {time_stats['prepare_batch']:.2f}s")
            print(f"  zero_grad_total         : {time_stats['zero_grad']:.2f}s")
            print(f"  forward_total           : {time_stats['forward']:.2f}s")
            print(f"  hier_loss_total         : {time_stats['hier_loss']:.2f}s")
            print(f"  aux_loss_total          : {time_stats['aux_loss']:.2f}s")
            print(f"  backward_total          : {time_stats['backward']:.2f}s")
            print(f"  optim_step_total        : {time_stats['optim_step']:.2f}s")
            print(f"  other_untracked_total   : {other_time:.2f}s")

            print("\n[Epoch Avg Per Batch]")
            print(f"  dataloader_wait_avg     : {avg_stats['dataloader_wait']:.2f}s")
            print(f"  unpack_batch_avg        : {avg_stats['unpack_batch']:.2f}s")
            print(f"  prepare_batch_avg       : {avg_stats['prepare_batch']:.2f}s")
            print(f"  zero_grad_avg           : {avg_stats['zero_grad']:.4f}s")
            print(f"  forward_avg             : {avg_stats['forward']:.2f}s")
            print(f"  hier_loss_avg           : {avg_stats['hier_loss']:.2f}s")
            print(f"  aux_loss_avg            : {avg_stats['aux_loss']:.2f}s")
            print(f"  backward_avg            : {avg_stats['backward']:.2f}s")
            print(f"  optim_step_avg          : {avg_stats['optim_step']:.2f}s")
            print(f"  batch_inside_avg        : {avg_stats['batch_total_inside']:.2f}s")

            if first_batch_stats is not None:
                print("\n[Batch 0 vs Avg]")
                print(f"  batch0_dataloader_wait  : {first_batch_stats['dataloader_wait']:.2f}s")
                print(f"  batch0_prepare_batch    : {first_batch_stats['prepare_batch']:.2f}s")
                print(f"  batch0_forward          : {first_batch_stats['forward']:.2f}s")
                print(f"  batch0_backward         : {first_batch_stats['backward']:.2f}s")
                print(f"  batch0_total_inside     : {first_batch_stats['batch_total_inside']:.2f}s")

        if self.local_rank == 0:
            timing_dir = os.path.join(self.ckpt_dir, "timing_logs")
            os.makedirs(timing_dir, exist_ok=True)
            timing_path = os.path.join(timing_dir, f"epoch_{epoch:03d}_timing.json")
            payload = {
                "epoch": int(epoch),
                "epoch_total_time": float(epoch_total_t),
                "num_batches": int(num_batches),
                "epoch_loss": float(epoch_loss),
                "totals": {k: float(v) for k, v in time_stats.items()},
                "averages": {k: float(v) for k, v in avg_stats.items()},
                "other_untracked_total": float(other_time),
                "first_batch": first_batch_stats,
            }
            with open(timing_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

        if self.save_epoch:
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, f'unicell_v1.ep{epoch}.pth'))

        return epoch_loss

    def predict(self, scDataset, batch_size):
        dataset = HMCNDataset(scDataset, input_type=self.input_type)
        old_dataset = self.dataset
        self.dataset = dataset

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.collate_fn)

        cls_outs = []
        labels = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if len(batch) == 5:
                    batch_data, batch_labels, cls_labels, tissue_labels, species_labels = batch
                    batch_batch_labels = None
                else:
                    batch_data, batch_labels, cls_labels, tissue_labels, species_labels, batch_batch_labels = batch

                if self.input_type == "GeneFormer":
                    base_model = self._base_model()
                    batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                    base_model.geneformer = base_model.geneformer.to(self.device)
                elif self.input_type == "scGPT":
                    batch_data = {
                        "input_ids": torch.stack([b["input_ids"] for b in batch_data]).to(torch.long).to(self.device),
                        "values": torch.stack([b["values"] for b in batch_data]).to(self.device)
                    }
                elif self.input_type == "expr":
                    batch_data = batch_data.to(self.device, non_blocking=True)
                else:
                    batch_data = torch.stack(batch_data).to(self.device)

                labels.extend(batch_labels.detach().cpu().numpy())

                with torch.cuda.amp.autocast(enabled=(self.input_type == "scGPT")):
                    _, _, _, global_cls_out, _, _ = self.model(batch_data)

                cls_outs.append(global_cls_out)

        self.dataset = old_dataset

        cls_layer_output = torch.cat(cls_outs, dim=0).detach().cpu().numpy()
        labels_pred = [self.cls2id[idx] for idx in np.argmax(cls_layer_output, axis=1)]
        cls_eval_res = compute_metrics_cls(labels, labels_pred)
        return cls_eval_res

    def collate_fn(self, batch):
        if len(batch[0]) == 5:
            batch_data, batch_labels, cls_labels, tissue_labels, species_labels = zip(*batch)
            batch_batch_labels = None
        else:
            batch_data, batch_labels, cls_labels, tissue_labels, species_labels, batch_batch_labels = zip(*batch)

        batch_labels = torch.stack(batch_labels)

        if self.input_type == "GeneFormer":
            base_model = self._base_model()
            model_input_size = get_model_input_size(base_model.geneformer)

            max_len = max(data["length"] for data in batch_data)

            input_data_minibatch = [torch.tensor(data["input_ids"], dtype=torch.long) for data in batch_data]
            pad_token_id = base_model.llm_vocab["<pad>"]
            input_data_minibatch = pad_tensor_list(
                input_data_minibatch, max_len, pad_token_id, model_input_size
            )

            new_batch_data = {
                "input_ids": input_data_minibatch,
                "length": torch.tensor([data["length"] for data in batch_data], dtype=torch.long)
            }

            if batch_batch_labels is None:
                return new_batch_data, batch_labels, cls_labels, list(tissue_labels), list(species_labels)
            else:
                return new_batch_data, batch_labels, cls_labels, list(tissue_labels), list(species_labels), torch.stack(batch_batch_labels)

        elif self.input_type == "expr":
            indices = list(batch_data)
            expr_batch = self.dataset.get_expr_batch(indices)

            if batch_batch_labels is None:
                return expr_batch, batch_labels, cls_labels, list(tissue_labels), list(species_labels)
            else:
                return expr_batch, batch_labels, cls_labels, list(tissue_labels), list(species_labels), torch.stack(batch_batch_labels)

        else:
            if batch_batch_labels is None:
                return batch_data, batch_labels, cls_labels, list(tissue_labels), list(species_labels)
            else:
                return batch_data, batch_labels, cls_labels, list(tissue_labels), list(species_labels), torch.stack(batch_batch_labels)
