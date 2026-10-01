#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/20 16:41
# @Author  : Luni Hu
# @File    : hmcn.py
# @Software: PyCharm

import json
from unicell.repo.scfoundation.load import load_model_frommmf
from unicell.repo.scfoundation.load import gatherData
from transformers import BertConfig, BertForMaskedLM
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

        # self.linear = nn.Sequential(
        #     nn.Linear(d_model, 512),
        #     nn.ReLU(),
        #     nn.LayerNorm(512),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.LayerNorm(256),
        #     nn.Linear(256, output_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(output_dim)
        # )
        
        self.linear = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        x = self.linear(x)
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
        self._decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class HMCN(nn.Module):
    """Implement HMCN(Hierarchical Multi-Label Classification Networks)
    Reference: "Hierarchical Multi-Label Classification Networks"
    """

    def __init__(
        self,
        input_type,
        input_dim,
        output_dim,
        num_classes,
        hierarchical_depth,
        global2local,
        hierarchical_class,
        hidden_layer_dropout,
        cls_num,
        tissue_cls_num,
        species_cls_num,
        llm_model_file,
        llm_vocab_file,
        llm_args_file,
        initialize_backbone_from_config=False,
    ):
        super(HMCN, self).__init__()

        self.input_type = input_type
        self.num_classes = num_classes
        self.hierarchical_depth = hierarchical_depth
        self.global2local = global2local
        self.hierarchical_class = hierarchical_class
        self.hidden_layer_dropout = hidden_layer_dropout

        self.local_layers = torch.nn.ModuleList()
        self.global_layers = torch.nn.ModuleList()

        if llm_vocab_file:
            self.llm_vocab = GeneVocab.from_file(llm_vocab_file)

        if llm_args_file:
            with open(llm_args_file, "r") as file:
                self.llm_args = json.load(file)

        if self.input_type == "scFoundation":
            self.scFoundation, self.scFoundation_config = load_model_frommmf(llm_model_file, "cell")
            output_dim = self.scFoundation_config["encoder"]["hidden_dim"]
            self.token_emb = self.scFoundation.token_emb
            self.pos_emb = self.scFoundation.pos_emb
            self.scf_encoder = self.scFoundation.encoder

            for na, param in self.scf_encoder.named_parameters():
                param.requires_grad = False
            for na, param in self.scf_encoder.transformer_encoder[-2].named_parameters():
                print("self.encoder.transformer_encoder ", na, " have grad")
                param.requires_grad = True
            self.norm = torch.nn.LayerNorm(self.scFoundation_config["encoder"]["hidden_dim"], eps=1e-6)

            input_dim = output_dim

        if self.input_type == "GeneFormer":
            if initialize_backbone_from_config:
                config = BertConfig.from_pretrained(llm_model_file)
                config.output_hidden_states = True
                config.output_attentions = False
                self.geneformer = BertForMaskedLM(config)
            else:
                self.geneformer = BertForMaskedLM.from_pretrained(
                    llm_model_file, output_hidden_states=True, output_attentions=False
                )
            output_dim = self.geneformer.config.hidden_size
            input_dim = output_dim

        if self.input_type == "scGPT":
            self.scgpt = load_gpt_model(
                self.llm_vocab,
                self.llm_args,
                llm_model_file,
                load_pretrained=not initialize_backbone_from_config,
            )
            output_dim = self.llm_args["embsize"]
            input_dim = output_dim

        # encoder output dim (x_encoded dim)
        self.input_dim = output_dim
        xenc_dim = output_dim  # <= NEW: explicit name for clarity

        self.encoder = Encoder(input_type=input_type, d_model=input_dim, output_dim=output_dim, dropout=0.2)

        for i in range(1, len(self.hierarchical_depth)):
            self.global_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.input_dim + self.hierarchical_depth[i - 1], self.hierarchical_depth[i]),
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(self.hierarchical_depth[i]),
                    torch.nn.Dropout(p=self.hidden_layer_dropout),
                )
            )
            self.local_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.hierarchical_depth[i], self.global2local[i]),
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(self.global2local[i]),
                    torch.nn.Linear(self.global2local[i], self.hierarchical_class[i]),
                )
            )

        self.global_layers.apply(self._init_weight)
        self.local_layers.apply(self._init_weight)

        self.linear = torch.nn.Linear(self.hierarchical_depth[-1], self.num_classes)
        self.linear.apply(self._init_weight)

        self.dropout = torch.nn.Dropout(p=self.hidden_layer_dropout)

        # ============================
        # Heads
        # - global_cls: still uses global_layer_activation (keep original behavior)
        # - tissue/species: CHANGED to use x_encoded instead of global_layer_activation
        # ============================
        if len(self.hierarchical_depth) < 2:
            raise ValueError("hierarchical_depth must have length >= 2.")

        global_head_dim = self.hierarchical_depth[-1]

        self.global_cls = nn.Linear(global_head_dim, cls_num)

        # CHANGED: tissue/species heads take x_encoded (dim = xenc_dim)
        self.tissue_cls = nn.Linear(xenc_dim, tissue_cls_num)
        self.species_cls = nn.Linear(xenc_dim, species_cls_num)

        if self.input_type == "scGPT":
            self.global_cls = ClsDecoder(global_head_dim, cls_num)
            self.tissue_cls = ClsDecoder(xenc_dim, tissue_cls_num)
            self.species_cls = ClsDecoder(xenc_dim, species_cls_num)
            self.linear = ClsDecoder(self.hierarchical_depth[-1], self.num_classes)

    def _init_weight(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.1)

    def get_parameter_optimizer_dict(self):
        params = super(HMCN, self).get_parameter_optimizer_dict()
        params.append({"params": self.local_layers.parameters()})
        params.append({"params": self.global_layers.parameters()})
        params.append({"params": self.linear.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """Update lr"""
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, x):
        if self.input_type == "scFoundation":
            value_labels = x > 0
            x, x_padding = gatherData(x, value_labels, self.scFoundation_config["pad_token_id"])
            data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
            position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.scFoundation_config["pad_token_id"])

            x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
            position_emb = self.pos_emb(position_gene_ids)
            x += position_emb

            x = self.scf_encoder(x, x_padding)

            x, _ = torch.max(x, dim=1)  # b,dim
            x = self.norm(x)

        if self.input_type == "GeneFormer":
            x = get_embs(self.geneformer, minibatch=x, emb_mode="cell")

        if self.input_type == "scGPT":
            src_key_padding_mask = x["input_ids"].eq(self.llm_vocab[self.llm_args["pad_token"]])
            output = self.scgpt._encode(x["input_ids"], x["values"], src_key_padding_mask)
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

        # celltype head keeps using global_layer_activation
        global_cls_output = self.global_cls(global_layer_activation)

        # CHANGED: tissue/species heads now use x_encoded
        tissue_cls_output = self.tissue_cls(x_encoded)
        species_cls_output = self.species_cls(x_encoded)

        return (
            global_layer_activation,
            global_layer_output,
            local_layer_outputs,
            global_cls_output,
            tissue_cls_output,
            species_cls_output,
        )


def load_gpt_model(vocab, args, model_file, load_pretrained=True):
    ntokens = len(vocab)
    model_param = {
        "ntoken": ntokens,
        "d_model": args["embsize"],
        "nhead": args["nheads"],
        "d_hid": args["d_hid"],
        "nlayers": args["nlayers"],
        "nlayers_cls": 3,
        "n_cls": 1,
        "dropout": 0.5,
        "pad_token": args["pad_token"],
        "do_mvc": False,
        "do_dab": False,
        "use_batch_labels": False,
        "num_batch_labels": None,
        "domain_spec_batchnorm": False,
        "input_emb_style": "continuous",
        "cell_emb_style": "cls",
        "mvc_decoder_style": "inner product",
        "ecs_threshold": 0.3,
        "explicit_zero_prob": False,
        "fast_transformer_backend": "flash",
        "pre_norm": False,
        "vocab": vocab,
        "pad_value": args["pad_value"],
        "n_input_bins": args["n_bins"],
        # Keep checkpoint key layout deterministic across machines. Flash-attn
        # checkpoints are converted below to PyTorch MHA in_proj weights.
        "use_fast_transformer": False,
    }
    for i in model_param:
        if i in args and i != "use_fast_transformer":
            model_param[i] = args[i]
    model = TransformerModel(**model_param)

    if not load_pretrained:
        return model
    if not model_file:
        raise ValueError("model_file is required when load_pretrained=True")

    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file, map_location="cpu")
    if not isinstance(pretrained_dict, dict):
        raise TypeError(
            "The scGPT checkpoint must contain a state-dict mapping, "
            f"got {type(pretrained_dict).__name__}."
        )
    if "model_state_dict" in pretrained_dict:
        pretrained_dict = pretrained_dict["model_state_dict"]
    elif "state_dict" in pretrained_dict:
        pretrained_dict = pretrained_dict["state_dict"]
    if not isinstance(pretrained_dict, dict):
        raise TypeError(
            "The scGPT checkpoint must contain a state-dict mapping, "
            f"got {type(pretrained_dict).__name__}."
        )

    compatible_dict = {}
    qkv_remapped_keys = {}
    shape_mismatches = {}
    for checkpoint_key, value in pretrained_dict.items():
        if not torch.is_tensor(value):
            continue

        normalized_key = checkpoint_key
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]

        candidate_keys = [normalized_key]
        if ".self_attn.Wqkv.weight" in normalized_key:
            candidate_keys.append(
                normalized_key.replace(
                    ".self_attn.Wqkv.weight", ".self_attn.in_proj_weight"
                )
            )
        elif ".self_attn.Wqkv.bias" in normalized_key:
            candidate_keys.append(
                normalized_key.replace(
                    ".self_attn.Wqkv.bias", ".self_attn.in_proj_bias"
                )
            )
        elif ".self_attn.in_proj_weight" in normalized_key:
            candidate_keys.append(
                normalized_key.replace(
                    ".self_attn.in_proj_weight", ".self_attn.Wqkv.weight"
                )
            )
        elif ".self_attn.in_proj_bias" in normalized_key:
            candidate_keys.append(
                normalized_key.replace(
                    ".self_attn.in_proj_bias", ".self_attn.Wqkv.bias"
                )
            )

        target_key = next(
            (
                key
                for key in candidate_keys
                if key in model_dict and value.shape == model_dict[key].shape
            ),
            None,
        )
        if target_key is not None:
            compatible_dict[target_key] = value
            if target_key != normalized_key:
                qkv_remapped_keys[checkpoint_key] = target_key
            continue

        existing_candidates = [key for key in candidate_keys if key in model_dict]
        if existing_candidates:
            shape_mismatches[checkpoint_key] = {
                key: tuple(model_dict[key].shape) for key in existing_candidates
            }

    if not compatible_dict:
        raise RuntimeError(
            "No compatible scGPT weights were found in "
            f"checkpoint {model_file!s}."
        )

    # The downstream UniCell heads may legitimately differ, but all parameters
    # used to produce the scGPT cell embedding must come from the checkpoint.
    critical_prefixes = ("encoder.", "value_encoder.", "transformer_encoder.")
    critical_keys = {
        key for key in model_dict if key.startswith(critical_prefixes)
    }
    missing_critical = sorted(critical_keys.difference(compatible_dict))
    if missing_critical:
        mismatch_details = []
        for checkpoint_key, target_shapes in shape_mismatches.items():
            if any(key in missing_critical for key in target_shapes):
                mismatch_details.append(
                    f"{checkpoint_key} {tuple(pretrained_dict[checkpoint_key].shape)} "
                    f"-> {target_shapes}"
                )
        detail = ""
        if mismatch_details:
            detail = " Shape mismatches: " + "; ".join(mismatch_details[:5])
        raise RuntimeError(
            "The scGPT checkpoint is missing or incompatible with critical "
            f"encoder weights ({len(missing_critical)} missing): "
            f"{', '.join(missing_critical[:8])}"
            f"{' ...' if len(missing_critical) > 8 else ''}.{detail}"
        )

    model.load_state_dict(compatible_dict, strict=False)
    print(
        "Loaded scGPT pretrained weights: "
        f"{len(compatible_dict)}/{len(pretrained_dict)} checkpoint tensors "
        f"({len(qkv_remapped_keys)} attention QKV tensors remapped)."
    )

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
    if keep_layers is None:
        keep_layers = []

    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for name, param in model.named_parameters():
        if isinstance(keep_layers, list):
            if name not in keep_layers:
                param.requires_grad = False
        else:
            raise ValueError("keep_layers should be a list of layer names.")

    ft_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total pretrain-model Params: {model_param_count}")
    print(f"Params for training after freezing: {ft_param_count}")

    return model
