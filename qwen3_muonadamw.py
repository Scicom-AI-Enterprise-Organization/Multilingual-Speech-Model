#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling
# task. Pointers for this are left as comments.

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import transformers
import random
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AddedToken,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    set_seed,
    get_wsd_schedule,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers import Qwen3ForCausalLM
import json
import numpy as np
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from cut_cross_entropy import linear_cross_entropy
from liger_kernel.transformers import apply_liger_kernel_to_qwen3, LigerFusedLinearCrossEntropyLoss

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

apply_liger_kernel_to_qwen3(
    rope=True,
    swiglu=True,
    rms_norm=True,
    cross_entropy=False,
    fused_linear_cross_entropy=False,
)

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " +
            ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None, metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")}, )
    config_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"}, )
    use_fast_tokenizer: bool = field(
        default=True, metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}, )
    model_revision: str = field(
        default="main", metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."}, )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine.")}, )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."),
            "choices": [
                "auto",
                "bfloat16",
                "float16",
                "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
                self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a text file)."})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

class Model(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.loss = LigerFusedLinearCrossEntropyLoss(reduction="sum")
        
    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, num_items_in_batch=None, **kwargs):
        super_out = self.model.forward(
            input_ids = input_ids, 
            position_ids = position_ids, 
            attention_mask = attention_mask, 
            output_hidden_states = True,
            **kwargs,
        )
        if labels is not None:
            embeddings = super_out.last_hidden_state
            embeddings = embeddings[:,:-1].reshape(-1, embeddings.shape[-1])
            labels = labels[..., 1:].contiguous()
            labels = labels.reshape(-1)
            loss = self.loss(self.lm_head.weight, embeddings, labels)
            num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
            return {'loss': loss}
        return super_out

class MuonPlusAdamW(torch.optim.Optimizer):
    """
    Hybrid optimizer: Muon for hidden layer 2D weights, AdamW for everything else.
    Based on Moonshot AI's "Muon is Scalable for LLM Training" (arXiv:2502.16982)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        # Muon hyperparameters
        muon_lr: Optional[float] = 1e-4,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.1,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        # AdamW hyperparameters
        adamw_betas: tuple = (0.9, 0.95),
        adamw_weight_decay: float = 0.01,
        adamw_eps: float = 1e-8,
        # Parameter filtering
        embed_patterns: tuple = ('embed', 'wte', 'wpe'),
        head_patterns: tuple = ('lm_head', 'head', 'output'),
    ):
        """
        Args:
            params: Model parameters
            lr: Base learning rate (used for AdamW, Muon uses muon_lr)
            muon_lr: Learning rate for Muon (default 1e-4)
            muon_momentum: Momentum for Muon
            muon_weight_decay: Weight decay for Muon (critical for scaling!)
            muon_nesterov: Use Nesterov momentum
            muon_ns_steps: Newton-Schulz iteration steps
            adamw_betas: Adam betas
            adamw_weight_decay: Weight decay for AdamW
            adamw_eps: Adam epsilon
            embed_patterns: Name patterns for embedding layers (use AdamW)
            head_patterns: Name patterns for output head layers (use AdamW)
        """
        if lr <= 0:
            raise ValueError("lr must be positive")

        # Convert to list of (name, param) if needed
        params = list(params)
        
        if params and isinstance(params[0], tuple):
            named_params = params
        else:
            named_params = [('', p) for p in params]

        muon_params, muon_params_name = [], []
        adamw_params, adamw_params_name = [], []

        embed_patterns: tuple = ('embed', 'wte', 'wpe')
        head_patterns: tuple = ('lm_head', 'head', 'output')

        for name, p in named_params:
            if not p.requires_grad:
                continue
            
            name_lower = name.lower()
            
            # Check if this is an embedding or head layer
            is_embed = any(pattern in name_lower for pattern in embed_patterns)
            is_head = any(pattern in name_lower for pattern in head_patterns)
            
            # Muon: only for 2D hidden layer weights (not embed/head)
            if p.ndim == 2 and not is_embed and not is_head:
                muon_params.append(p)
                muon_params_name.append(name)
            else:
                adamw_params.append(p)
                adamw_params_name.append(name)

        # Default Muon LR is 0.02 (paper recommendation)
        effective_muon_lr = muon_lr if muon_lr is not None else 0.02

        param_groups = [
            {"params": muon_params, "type": "muon", "lr": effective_muon_lr},
            {"params": adamw_params, "type": "adamw", "lr": lr},
        ]

        defaults = {"lr": lr}
        super().__init__(param_groups, defaults)

        # Store param counts for logging
        self.muon_param_count = sum(p.numel() for p in muon_params)
        self.adamw_param_count = sum(p.numel() for p in adamw_params)

        # Initialize sub-optimizers
        if muon_params:
            self._muon = torch.optim.Muon(
                muon_params,
                lr=effective_muon_lr,
                momentum=muon_momentum,
                weight_decay=muon_weight_decay,
                nesterov=muon_nesterov,
                ns_steps=muon_ns_steps,
            )
        else:
            self._muon = None

        if adamw_params:
            self._adamw = torch.optim.AdamW(
                adamw_params,
                lr=lr,
                betas=adamw_betas,
                weight_decay=adamw_weight_decay,
                eps=adamw_eps,
            )
        else:
            self._adamw = None

    def __repr__(self):
        return (
            f"MuonPlusAdamW(\n"
            f"  muon_params: {self.muon_param_count:,}\n"
            f"  adamw_params: {self.adamw_param_count:,}\n"
            f")"
        )

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._adamw is not None:
            self._adamw.step()
        if self._muon is not None:
            self._muon.step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none)
        if self._muon is not None:
            self._muon.zero_grad(set_to_none)

    def state_dict(self):
        """Return combined state dict."""
        return {
            'muon': self._muon.state_dict() if self._muon else None,
            'adamw': self._adamw.state_dict() if self._adamw else None,
        }

    def load_state_dict(self, state_dict):
        """Load combined state dict."""
        if self._muon is not None and state_dict.get('muon'):
            self._muon.load_state_dict(state_dict['muon'])
        if self._adamw is not None and state_dict.get('adamw'):
            self._adamw.load_state_dict(state_dict['adamw'])
        
def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    extra = [AddedToken('<|speech_start|>')]
    for i in range(65536):
        extra.append(AddedToken(f'<|s_{i}|>'))
    tokenizer.add_tokens(extra)
    
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    min_dtype = torch.finfo(torch_dtype).min
    sequence_length = data_args.block_size

    class UInt32(Encoding):
        def encode(self, obj) -> bytes:
            return obj.tobytes()

        def decode(self, data: bytes):
            return np.frombuffer(data, np.uint32)

    _encodings['uint32'] = UInt32

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, local):
            self.dataset = LocalDataset(local=local)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            data.pop('audio', None)
            data.pop('text', None)
            data.pop('token_type_ids', None)

            if data['attention_mask'].max() > sequence_length:
                print(data)
                return

            for k in data.keys():
                data[k] = data[k].astype(np.int64)
        
            return data

        def __len__(self):
            return len(self.dataset)

    model = Model.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation = 'kernels-community/vllm-flash-attn3',
        torch_dtype = model_args.torch_dtype,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False, pad_to_multiple_of=8)
    print(model)

    dataset = DatasetFixed(data_args.train_file)
    print('dataset', len(dataset), dataset[0]['attention_mask'].shape)

    def collator(batch):
        batch = [b for b in batch if b is not None]
        input_ids = [b['input_ids'] for b in batch]
        position_ids = [b['position_ids'] for b in batch]
        labels = [b['input_ids'].copy() for b in batch]
        attention_mask = [b['attention_mask'] for b in batch]
        input_ids = np.concatenate(input_ids)
        position_ids = np.concatenate(position_ids)
        labels = np.concatenate(labels)
        query_lens = np.concatenate(attention_mask)
        cumsum = [0] + np.cumsum(query_lens).tolist()
        max_cumsum = int(np.max(cumsum))
        cu_seq_lens_q = torch.tensor(cumsum, dtype=torch.int32)
        cu_seq_lens_k = torch.tensor(cumsum, dtype=torch.int32)
        max_seqlen_q = np.max(query_lens)
        return {
            'input_ids': torch.tensor(input_ids)[None],
            'position_ids': torch.tensor(position_ids)[None],
            'labels': torch.tensor(labels)[None],
            'cu_seq_lens_q': cu_seq_lens_q,
            'cu_seq_lens_k': cu_seq_lens_k,
            'max_length_q': max_seqlen_q,
            'max_length_k': max_seqlen_q
        }

    optimizer = MuonPlusAdamW(model.named_parameters(), lr=training_args.learning_rate)
    len_dataset = math.ceil(len(dataset) / torch.cuda.device_count())
    len_dataloader = math.ceil(len_dataset / training_args.per_device_train_batch_size)
    num_update_steps_per_epoch = max(
        len_dataloader // training_args.gradient_accumulation_steps
        + int(len_dataloader % training_args.gradient_accumulation_steps > 0),
        1,
    )
    max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    print('max_steps', max_steps)
    lr_scheduler = get_wsd_schedule(
        optimizer, 
        num_warmup_steps=training_args.warmup_steps,
        num_decay_steps=training_args.lr_scheduler_kwargs['num_decay_steps'],
        num_training_steps=max_steps,
        min_lr_ratio=training_args.lr_scheduler_kwargs['min_lr_ratio'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
        optimizers=(optimizer, lr_scheduler),
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()