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
Optimizer-search trainer: one short-run trainer with a pluggable optimizer, driven by
hyperparameter_search.py. Supersedes qwen3_muonadamw_search.py (`--optimizer muon`
reproduces it exactly).

Optimizers:
  adamw               torch.optim.AdamW on everything
  muon                torch.optim.Muon on 2D hidden weights + AdamW on embeddings/head/rest
  shampoo             pytorch_optimizer.ScalableShampoo on 2D hidden weights + AdamW on rest
  soap                pytorch_optimizer.SOAP on 2D hidden weights + AdamW on rest
  lion                pytorch_optimizer.Lion on everything
  ademamix            pytorch_optimizer.AdEMAMix on everything

Hybrid modes take `--matrix_lr` for the 2D-hidden-weight sub-optimizer while
`--learning_rate` drives the AdamW side (and everything for full-model optimizers).
Weight decay comes from `--weight_decay`; norm/bias parameters never decay.
"""

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
from chinidataset import StreamingDataset
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
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Packed dataset to validate on — the dev-split pack. Runs are ranked "
                          "on this when it is set."})
    max_eval_blocks: int = field(
        default=256,
        metadata={"help": "Evaluate on an evenly-strided subset of --validation_file this many "
                          "blocks wide (0 = all of it). A full dev pass per eval costs more than "
                          "the 100-step run itself."})
    audio_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Root the pack's `audio` paths resolve against. Setting it switches the "
                          "trainer to the raw log-mel front end (mel_audio.py) — the pack must be "
                          "a mel pack, which stores paths instead of speech tokens."})
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


@dataclass
class SearchArguments:
    """Search-only knobs on top of TrainingArguments."""

    optimizer: str = field(
        default='adamw',
        metadata={"help": "adamw | muon | shampoo | soap | lion | ademamix"},
    )
    matrix_lr: Optional[float] = field(
        default=None,
        metadata={"help": "LR for the 2D-hidden-weight sub-optimizer in hybrid modes (muon/shampoo/soap)."},
    )
    num_decay_steps: int = field(
        default=243, metadata={"help": "WSD scheduler decay steps."})
    min_lr_ratio: float = field(
        default=0.1, metadata={"help": "WSD scheduler final LR ratio."})
    added_tokens_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "JSON list of extra tokens to append AFTER the 65,537 speech tokens, in order "
                "(e.g. the <|STT|> + language tokens written by the STT multipacking). Must be "
                "the same file the data was packed with, or token ids will not line up."
            )
        },
    )


class Model(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.loss = LigerFusedLinearCrossEntropyLoss(reduction="sum")

    def compute_loss(self, super_out, labels, num_items_in_batch):
        embeddings = super_out.last_hidden_state
        embeddings = embeddings[:, :-1].reshape(-1, embeddings.shape[-1])
        labels = labels[..., 1:].contiguous().reshape(-1)
        loss = self.loss(self.lm_head.weight, embeddings, labels)
        if num_items_in_batch is None:
            # evaluation_loop does not pass it; count the tokens that carry a label
            num_items_in_batch = (labels != -100).sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        return loss / num_items_in_batch

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, num_items_in_batch=None, **kwargs):
        super_out = self.model.forward(
            input_ids = input_ids,
            position_ids = position_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
            **kwargs,
        )
        if labels is not None:
            return {'loss': self.compute_loss(super_out, labels, num_items_in_batch)}
        return super_out


def build_mel_model_class():
    """Model with the raw log-mel front end spliced in — built only for mel packs.

    Kept behind a factory so the token-only arms neither import mel_audio nor carry the
    projector's parameters (which would land in the optimizer's param groups and make the
    arms un-comparable).
    """
    from mel_audio import MelProjector, WhisperMel

    class MelModel(Model):
        def __init__(self, config):
            super().__init__(config)
            self.mel = WhisperMel()
            self.mel_projector = MelProjector(config.hidden_size)

        def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None,
                    mel_waveform=None, mel_frame_index=None, mel_segment_id=None,
                    mel_num_segments=0, num_items_in_batch=None, **kwargs):
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if mel_waveform is not None and mel_waveform.numel():
                # fp32 throughout: a bf16 STFT loses the quiet end of the spectrum, and
                # whisper's (max - 8) floor then spreads that loss over the utterance
                with torch.autocast(device_type=inputs_embeds.device.type, enabled=False):
                    mel = self.mel(mel_waveform.float(), mel_frame_index, mel_segment_id,
                                   int(mel_num_segments))
                features = self.mel_projector(mel)
                slots = input_ids == self.config.mel_token_id
                if int(slots.sum()) != features.shape[0]:
                    raise RuntimeError(
                        f'{int(slots.sum())} <|mel|> placeholders but {features.shape[0]} mel '
                        'positions — the pack and the mel front end disagree on the frame rate')
                inputs_embeds = inputs_embeds.masked_scatter(
                    slots.unsqueeze(-1), features.to(inputs_embeds.dtype))
                mel_touch = None
            else:
                # no mel document in this micro-batch — keep the projector in the graph or
                # ddp_find_unused_parameters=false stalls the all-reduce
                mel_touch = self.mel_projector.zero_probe(inputs_embeds)

            super_out = self.model.forward(
                inputs_embeds = inputs_embeds,
                position_ids = position_ids,
                attention_mask = attention_mask,
                output_hidden_states = True,
                **kwargs,
            )
            if labels is not None:
                loss = self.compute_loss(super_out, labels, num_items_in_batch)
                if mel_touch is not None:
                    loss = loss + mel_touch.to(loss.dtype)
                return {'loss': loss}
            return super_out

    return MelModel


def split_matrix_params(named_params):
    """2D hidden-layer weights vs everything else (embeddings, head, norms, biases).

    The matrix side is what Muon/Shampoo/SOAP precondition; the rest stays on AdamW —
    the split used by Moonshot's "Muon is Scalable for LLM Training" (arXiv:2502.16982).
    """
    embed_patterns = ('embed', 'wte', 'wpe')
    head_patterns = ('lm_head', 'head', 'output')

    matrix, matrix_names, rest, rest_names = [], [], [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        name_lower = name.lower()
        is_embed = any(pattern in name_lower for pattern in embed_patterns)
        is_head = any(pattern in name_lower for pattern in head_patterns)
        if p.ndim == 2 and not is_embed and not is_head:
            matrix.append(p)
            matrix_names.append(name)
        else:
            rest.append(p)
            rest_names.append(name)
    return matrix, matrix_names, rest, rest_names


def decay_param_groups(named_params, lr, weight_decay):
    """HF-style grouping: no weight decay on 1D params (norms, biases)."""
    decay = [p for n, p in named_params if p.requires_grad and p.ndim >= 2]
    no_decay = [p for n, p in named_params if p.requires_grad and p.ndim < 2]
    return [
        {'params': decay, 'lr': lr, 'weight_decay': weight_decay},
        {'params': no_decay, 'lr': lr, 'weight_decay': 0.0},
    ]


class HybridOptimizer(torch.optim.Optimizer):
    """Matrix optimizer (Muon/Shampoo/SOAP) on 2D hidden weights + AdamW on everything else.

    Generalization of the MuonPlusAdamW hybrid from qwen3_muonadamw.py; the
    step/state_dict/param_groups plumbing keeps HF Trainer + LambdaLR schedulers happy.
    """

    def __init__(
        self,
        named_params,
        matrix_factory,          # callable(params, lr, weight_decay) -> torch.optim.Optimizer
        lr: float = 1e-4,        # AdamW side
        matrix_lr: float = 1e-3,
        weight_decay: float = 0.1,
        adamw_betas: tuple = (0.9, 0.999),
        adamw_eps: float = 1e-8,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")

        named_params = list(named_params)
        matrix_params, matrix_names, rest_params, rest_names = split_matrix_params(named_params)
        print('matrix_params_name', matrix_names)
        print('adamw_params_name', rest_names)

        param_groups = [
            {"params": matrix_params, "type": "matrix", "lr": matrix_lr},
            {"params": rest_params, "type": "adamw", "lr": lr},
        ]
        self._matrix = None
        self._adamw = None
        super().__init__(param_groups, {"lr": lr})

        self.matrix_param_count = sum(p.numel() for p in matrix_params)
        self.adamw_param_count = sum(p.numel() for p in rest_params)

        if matrix_params:
            self._matrix = matrix_factory(matrix_params, matrix_lr, weight_decay)
        if rest_params:
            self._adamw = torch.optim.AdamW(
                rest_params,
                lr=lr,
                betas=adamw_betas,
                weight_decay=weight_decay,
                eps=adamw_eps,
            )

    def __repr__(self):
        return (
            f"HybridOptimizer(\n"
            f"  matrix: {type(self._matrix).__name__} {self.matrix_param_count:,} params\n"
            f"  adamw: {self.adamw_param_count:,} params\n"
            f")"
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._adamw is not None:
            self._adamw.step()
        if self._matrix is not None:
            self._matrix.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none)
        if self._matrix is not None:
            self._matrix.zero_grad(set_to_none)

    def state_dict(self):
        return {
            'matrix': self._matrix.state_dict() if self._matrix else None,
            'adamw': self._adamw.state_dict() if self._adamw else None,
        }

    def load_state_dict(self, state_dict):
        if self._matrix is not None and state_dict.get('matrix'):
            self._matrix.load_state_dict(state_dict['matrix'])
        if self._adamw is not None and state_dict.get('adamw'):
            self._adamw.load_state_dict(state_dict['adamw'])

    @property
    def param_groups(self):
        if not hasattr(self, '_matrix'):
            return self.__dict__.get('param_groups', [])
        if self._matrix is None and self._adamw is None:
            return self.__dict__.get('param_groups', [])
        groups = []
        if self._matrix is not None:
            groups.extend(self._matrix.param_groups)
        if self._adamw is not None:
            groups.extend(self._adamw.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value):
        # managed by the sub-optimizers
        pass


def build_optimizer(model, search_args, training_args):
    name = search_args.optimizer.lower()
    lr = training_args.learning_rate
    wd = training_args.weight_decay
    matrix_lr = search_args.matrix_lr

    hybrid = name in ('muon', 'shampoo', 'soap')
    if hybrid and matrix_lr is None:
        raise ValueError(f'--matrix_lr is required for optimizer={name}')

    if name == 'adamw':
        return torch.optim.AdamW(
            decay_param_groups(model.named_parameters(), lr, wd),
            lr=lr, betas=(0.9, 0.999), eps=1e-8,
        )

    if name == 'muon':
        def factory(params, mlr, decay):
            return torch.optim.Muon(
                params, lr=mlr, momentum=0.95, weight_decay=decay,
                nesterov=True, ns_steps=5,
            )
    elif name == 'shampoo':
        from pytorch_optimizer import ScalableShampoo

        def factory(params, mlr, decay):
            # preconditioning_compute_steps default (1000) would never refresh the
            # preconditioner inside a 100-step search run
            return ScalableShampoo(
                params, lr=mlr, betas=(0.9, 0.999), weight_decay=decay,
                decoupled_weight_decay=True,
                start_preconditioning_step=10,
                preconditioning_compute_steps=10,
                statistics_compute_steps=1,
            )
    elif name == 'soap':
        from pytorch_optimizer import SOAP

        def factory(params, mlr, decay):
            return SOAP(
                params, lr=mlr, betas=(0.95, 0.95), weight_decay=decay,
                precondition_frequency=10,
            )
    elif name == 'lion':
        from pytorch_optimizer import Lion
        return Lion(
            decay_param_groups(model.named_parameters(), lr, wd),
            lr=lr, betas=(0.9, 0.99), weight_decouple=True,
        )
    elif name == 'ademamix':
        from pytorch_optimizer import AdEMAMix
        return AdEMAMix(
            decay_param_groups(model.named_parameters(), lr, wd),
            lr=lr, betas=(0.9, 0.999, 0.9999), alpha=5.0, weight_decouple=True,
        )
    else:
        raise ValueError(f'unknown optimizer {name!r}')

    return HybridOptimizer(
        model.named_parameters(),
        matrix_factory=factory,
        lr=lr,
        matrix_lr=matrix_lr,
        weight_decay=wd,
    )


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SearchArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, search_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, search_args, training_args = parser.parse_args_into_dataclasses()

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
    logger.info(f"Search parameters {search_args}")

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
    if search_args.added_tokens_file:
        with open(search_args.added_tokens_file) as f:
            added_tokens = json.load(f)
        tokenizer.add_tokens([AddedToken(t) for t in added_tokens])
        logger.info(f'appended {len(added_tokens)} tokens from {search_args.added_tokens_file}')

    mel = bool(data_args.audio_dir)
    mel_token_id, mel_label_ids = None, None
    if mel:
        from mel_audio import MEL_TOKENS, layout_segments, unpack_block

        if not os.path.isdir(data_args.audio_dir):
            raise SystemExit(f'--audio_dir {data_args.audio_dir} does not exist')
        mel_token_id = tokenizer.convert_tokens_to_ids('<|mel|>')
        if mel_token_id == tokenizer.unk_token_id or mel_token_id is None:
            raise SystemExit('--audio_dir is set but <|mel|> is not in the tokenizer — point '
                             '--added_tokens_file at the mel pack\'s added-tokens JSON')
        # mel tokens are inputs the model is fed, never outputs it should emit
        mel_label_ids = np.array(tokenizer.convert_tokens_to_ids(MEL_TOKENS), dtype=np.int64)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    min_dtype = torch.finfo(torch_dtype).min
    sequence_length = data_args.block_size

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, local, indices=None):
            self.dataset = StreamingDataset(local=local)
            # evenly-strided view, for capping the dev pass
            self.indices = indices

        def __getitem__(self, idx):
            row = self.dataset[int(self.indices[idx]) if self.indices is not None else idx]
            if mel:
                return unpack_block(row, sequence_length, data_args.audio_dir)

            # copy: the reader hands back the same dict object for a repeated index, so
            # popping off it would strip those columns from every later read
            data = dict(row)
            data.pop('audio', None)
            data.pop('text', None)
            data.pop('token_type_ids', None)

            for k in data.keys():
                data[k] = np.asarray(data[k]).astype(np.int64)

            if data['attention_mask'].max() > sequence_length:
                print(data)
                return

            return data

        def __len__(self):
            return len(self.indices) if self.indices is not None else len(self.dataset)

    model_cls = build_mel_model_class() if mel else Model
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation = 'kernels-community/vllm-flash-attn3',
        torch_dtype = model_args.torch_dtype,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False, pad_to_multiple_of=8)
    if mel:
        model.config.mel_token_id = mel_token_id
        # a submodule built in __init__ comes back through from_pretrained's missing-keys
        # path; print it so a zeroed or nan projector shows up here, not as a 0.0 loss
        fc1 = model.mel_projector.fc1.weight
        print(f'mel_projector.fc1: std={float(fc1.std()):.5f} requires_grad={fc1.requires_grad}')
    print(model)

    dataset = DatasetFixed(data_args.train_file)
    print('dataset', len(dataset), dataset[0]['attention_mask'].shape)

    eval_dataset = None
    if data_args.validation_file:
        eval_dataset = DatasetFixed(data_args.validation_file)
        total = len(eval_dataset)
        cap = data_args.max_eval_blocks
        if cap and total > cap:
            # strided, not a prefix: packs are written worker by worker, so the first N
            # blocks are one slice of the corpus rather than a sample of it
            eval_dataset.indices = np.linspace(0, total, cap, endpoint=False).astype(np.int64)
        print('eval dataset', len(eval_dataset), f'of {total} blocks')

    def collator(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            # only reachable if a single document is longer than block_size
            raise RuntimeError('every block in this micro-batch was dropped as oversized')
        input_ids = [b['input_ids'] for b in batch]
        position_ids = [b['position_ids'] for b in batch]
        labels = [b['input_ids'].copy() for b in batch]
        attention_mask = [b['attention_mask'] for b in batch]
        input_ids = np.concatenate(input_ids)
        position_ids = np.concatenate(position_ids)
        labels = np.concatenate(labels)
        if mel:
            labels[np.isin(labels, mel_label_ids)] = -100
        query_lens = np.concatenate(attention_mask)
        cumsum = [0] + np.cumsum(query_lens).tolist()
        max_cumsum = int(np.max(cumsum))
        cu_seq_lens_q = torch.tensor(cumsum, dtype=torch.int32)
        cu_seq_lens_k = torch.tensor(cumsum, dtype=torch.int32)
        max_seqlen_q = np.max(query_lens)
        out = {
            'input_ids': torch.tensor(input_ids)[None],
            'position_ids': torch.tensor(position_ids)[None],
            'labels': torch.tensor(labels)[None],
            'cu_seq_lens_q': cu_seq_lens_q,
            'cu_seq_lens_k': cu_seq_lens_k,
            'max_length_q': max_seqlen_q,
            'max_length_k': max_seqlen_q
        }
        if mel:
            # utterance order across the flattened batch has to match the order the
            # <|mel|> placeholders appear in
            waveforms = [w for b in batch for w in b.get('waveforms', ())]
            if waveforms:
                buffer, frame_index, segment_id = layout_segments(waveforms)
                out['mel_waveform'] = torch.from_numpy(buffer)
                out['mel_frame_index'] = torch.from_numpy(frame_index)
                out['mel_segment_id'] = torch.from_numpy(segment_id)
                out['mel_num_segments'] = len(waveforms)
        return out

    optimizer = build_optimizer(model, search_args, training_args)
    print(optimizer)

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
        num_decay_steps=search_args.num_decay_steps,
        num_training_steps=max_steps,
        min_lr_ratio=search_args.min_lr_ratio,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
        optimizers=(optimizer, lr_scheduler),
    )

    trainer.train()

    # search runs are shorter than save_steps, so persist the loss curve explicitly
    # for hyperparameter_search.py to rank runs
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(os.path.join(training_args.output_dir, 'trainer_state.json'))

    print('final param group LRs:', [g['lr'] for g in optimizer.param_groups])


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
