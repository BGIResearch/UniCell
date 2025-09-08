#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/20 16:41
# @Author  : Luni Hu
# @File    : hmcn
# @Software: PyCharm

import json
from unicell.repo.scfoundation.load import load_model_frommmf
from unicell.repo.scfoundation.load import gatherData
from transformers import BertForMaskedLM
from unicell.repo.geneformer.emb_extractor_catree import get_embs
from unicell.repo.scgpt.tokenizer.gene_tokenizer import GeneVocab
from unicell.repo.scgpt.model import TransformerModel

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_type, d_model, output_dim=128, dropout=0.1):
        super(Encoder, self).__init__()

        self.input_type = input_type
        if self.input_type == "scGPT":
            dropout = 0
        self.dropout = nn.Dropout(dropout)

        # Define the linear layers with intermediate operations
        self.linear = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # Pass through the defined layers
        x = self.linear(x)
        # Apply dropout if the input type is "expr"
        if self.input_type == "expr":
            x = self.dropout(x)
        return x

class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
            self,
            d_model: int,
            n_cls: int,
            nlayers: int = 3,
            activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x) :
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class HMCN(nn.Module):
    """ Implement HMCN(Hierarchical Multi-Label Classification Networks)
        Reference: "Hierarchical Multi-Label Classification Networks"
    """

    def __init__(self, input_type, input_dim, output_dim, num_classes, hierarchical_depth, global2local, hierarchical_class, hidden_layer_dropout, cls_num, llm_model_file, llm_vocab_file, llm_args_file):
        super(HMCN, self).__init__()

        self.input_type = input_type
        self.num_classes = num_classes
        self.hierarchical_depth = hierarchical_depth
        self.global2local = global2local
        self.hierarchical_class = hierarchical_class
        self.hidden_layer_dropout = hidden_layer_dropout

        self.local_layers = torch.nn.ModuleList()
        self.global_layers = torch.nn.ModuleList()
        # TODO: add the param for the output_dim

        if llm_vocab_file:
            self.llm_vocab = GeneVocab.from_file(llm_vocab_file)

        if llm_args_file:
            with open(llm_args_file, 'r') as file:
                self.llm_args = json.load(file)

        if self.input_type == "scFoundation":
            self.scFoundation, self.scFoundation_config = load_model_frommmf(llm_model_file, "cell")
            output_dim = self.scFoundation_config['encoder']['hidden_dim']
            self.token_emb = self.scFoundation.token_emb
            self.pos_emb = self.scFoundation.pos_emb
            self.scf_encoder = self.scFoundation.encoder

            for na, param in self.scf_encoder.named_parameters():
                param.requires_grad = False
            for na, param in self.scf_encoder.transformer_encoder[-2].named_parameters():
                print('self.encoder.transformer_encoder ', na, ' have grad')
                param.requires_grad = True
            self.norm = torch.nn.LayerNorm(self.scFoundation_config['encoder']['hidden_dim'], eps=1e-6)

            input_dim = output_dim

        if self.input_type == "GeneFormer":
            self.geneformer = BertForMaskedLM.from_pretrained(llm_model_file,
                                                              output_hidden_states=True,
                                                              output_attentions=False)

            output_dim = self.geneformer.config.hidden_size
            input_dim = output_dim

        if self.input_type == "scGPT":
            self.scgpt = load_gpt_model(self.llm_vocab, self.llm_args, llm_model_file)

            output_dim = self.llm_args["embsize"]
            input_dim = output_dim

        self.input_dim = output_dim

        self.encoder = Encoder(input_type=input_type, d_model=input_dim, output_dim=output_dim, dropout=0.2)

        for i in range(1, len(self.hierarchical_depth)):
            self.global_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.input_dim + self.hierarchical_depth[i - 1], self.hierarchical_depth[i]),
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(self.hierarchical_depth[i]),
                    torch.nn.Dropout(p=self.hidden_layer_dropout)
                ))
            self.local_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.hierarchical_depth[i], self.global2local[i]),
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(self.global2local[i]),
                    torch.nn.Linear(self.global2local[i], self.hierarchical_class[i])
                ))

        self.global_layers.apply(self._init_weight)
        self.local_layers.apply(self._init_weight)
        self.linear = torch.nn.Linear(self.hierarchical_depth[-1], self.num_classes)
        self.linear.apply(self._init_weight)
        self.dropout = torch.nn.Dropout(p=self.hidden_layer_dropout)
        self.global_cls = nn.Linear(self.hierarchical_depth[i - 1], cls_num)

        if self.input_type == "scGPT":
            self.global_cls = ClsDecoder(self.hierarchical_depth[i - 1], cls_num)
            self.linear = ClsDecoder(self.hierarchical_depth[-1], self.num_classes)


    def _init_weight(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.1)



    def get_parameter_optimizer_dict(self):
        params = super(HMCN, self).get_parameter_optimizer_dict()
        params.append({'params': self.local_layers.parameters()})
        params.append({'params': self.global_layers.parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """ Update lr
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, x):

        if self.input_type == "scFoundation":

            value_labels = x > 0
            x, x_padding = gatherData(x, value_labels, self.scFoundation_config['pad_token_id'])
            data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
            position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                              self.scFoundation_config['pad_token_id'])

            x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
            position_emb = self.pos_emb(position_gene_ids)
            x += position_emb

            x = self.scf_encoder(x, x_padding)

            # mlp
            x, _ = torch.max(x, dim=1)  # b,dim

            x = self.norm(x)

        if self.input_type == "GeneFormer":

            x = get_embs(self.geneformer, minibatch=x, emb_mode="cell")

        if self.input_type == "scGPT":

            src_key_padding_mask = x["input_ids"].eq(self.llm_vocab[self.llm_args["pad_token"]])
            output = self.scgpt._encode(x["input_ids"],
                                        x["values"],
                                        src_key_padding_mask)
            x = self.scgpt._get_cell_emb_from_layer(output, x["values"])

        x_encoded = self.encoder(x)

        local_layer_outputs = []
        global_layer_activation = x_encoded

        for i, (local_layer, global_layer) in enumerate(zip(self.local_layers, self.global_layers)):
            local_layer_activation = global_layer(global_layer_activation)
            local_layer_output = local_layer(local_layer_activation)
            if not self.input_type == "scGPT":
                local_layer_output = torch.sigmoid(local_layer_output)
            local_layer_outputs.append(local_layer_output)
            if i < len(self.global_layers) - 1:
                global_layer_activation = torch.cat((local_layer_activation, x_encoded), 1)
            else:
                global_layer_activation = local_layer_activation
        global_layer_output = self.linear(global_layer_activation)
        if not self.input_type == "scGPT":
            global_layer_output = torch.sigmoid(global_layer_output)
        global_cls_output = self.global_cls(global_layer_activation)

        return global_layer_activation, global_layer_output, local_layer_outputs, global_cls_output

def load_gpt_model(vocab, args, model_file):
    ntokens = len(vocab)
    model_param = {
        'ntoken': ntokens,
        'd_model': args["embsize"],
        'nhead': args["nheads"],
        'd_hid': args["d_hid"],
        'nlayers': args["nlayers"],
        'nlayers_cls': 3,
        'n_cls': 1,
        'dropout': 0.5,
        'pad_token': "<pad>",
        'do_mvc': False,
        'do_dab': False,
        'use_batch_labels': False,
        'num_batch_labels': None,
        'domain_spec_batchnorm': False,
        'input_emb_style': "continuous",
        'cell_emb_style': "cls",
        'mvc_decoder_style': "inner product",
        'ecs_threshold': 0.3,
        'explicit_zero_prob': False,
        'fast_transformer_backend': "flash",
        'pre_norm': False,
        'vocab': vocab,
        'pad_value': args["pad_value"],
        'n_input_bins': args["n_bins"],
        'use_fast_transformer': True,
    }
    for i in model_param:
        if i in args:
            model_param[i] = args[i]
    model = TransformerModel(**model_param)

    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file, map_location='cpu')
    if 'model_state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_state_dict']
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def freezon_model(model, keep_layers=None):
    """
    Freezes model parameters except for the layers specified in `keep_layers`.

    Parameters:
    - model: The model to be frozen.
    - keep_layers: A list of layer names or indices to keep trainable.

    Returns:
    - The model with frozen layers.
    """
    # If no keep_layers argument is provided, freeze all encoder layers
    if keep_layers is None:
        keep_layers = []

    # Count total parameters before freezing
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Freeze all layers except those in `keep_layers`
    for name, param in model.named_parameters():
        # Freeze parameters if they are not in `keep_layers`
        if isinstance(keep_layers, list):
            if name not in keep_layers:
                param.requires_grad = False
        else:
            raise ValueError("keep_layers should be a list of layer names.")

    # Count the number of trainable parameters after freezing
    ft_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Log the results
    print(f"Total pretrain-model Params: {model_param_count}")
    print(f"Params for training after freezing: {ft_param_count}")

    return model
