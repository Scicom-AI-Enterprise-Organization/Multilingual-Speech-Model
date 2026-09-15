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
"""Qwen3 speech trainer with a raw log-mel input path, for the three-task iteration:

    1. TTS, audio tokenizer   <|im_start|>{speaker}: {text}<|speech_start|>{<|s_N|>}<|im_end|>
    2. STT, audio tokenizer   <|im_start|><|STT|>{<|s_N|>}<|{lang}|>{text}<|im_end|>
    3. STT, raw mel           <|im_start|><|STT|><|mel_start|>{<|mel|>}<|mel_end|><|{lang}|>{text}<|im_end|>

Tasks 1 and 2 are unchanged token streams, so their existing packs are reused as-is.
Task 3 carries audio in the pack; `<|mel|>` is a placeholder whose embedding is replaced
by a projected whisper log-mel frame pair (see mel_audio.py). All three are mixed at
train time via `--train_file "dirA:1.0,dirB:1.0,dirC:2.0"`, so the ratio is a launch
flag and nothing has to be repacked to change it.

AdamW rather than Muon+AdamW because AdamW won the one-epoch ablation (see README).
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
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers import Qwen3ForCausalLM
import json
import numpy as np
from chinidataset import StreamingDataset
from cut_cross_entropy import linear_cross_entropy
from liger_kernel.transformers import apply_liger_kernel_to_qwen3, LigerFusedLinearCrossEntropyLoss

from mel_audio import (
    MEL_TOKENS,
    MelProjector,
    WhisperMel,
    build_speech_tokenizer,
    load_segments,
    unpack_block,
    layout_segments,
    mixture_index,
    parse_train_files,
)

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
    stt_tokens_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to an STT pack's stt_added_tokens.json (<|STT|> + the language tokens). "
                "Required for the STT tasks: these ids must be added in the same order the pack "
                "used, right after the 65,536 <|s_N|> tokens."
            )
        },
    )
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated packed dataset directories, each optionally suffixed with a "
                "sampling weight: 'tts_dir:1.0,stt_dir:1.0,mel_dir:2.0'. The weight is how many "
                "epochs of that dataset go into one training epoch (1.0 = plain concat)."
            )
        },
    )
    audio_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Root of the extracted audio tree the raw-mel pack's paths are relative to. "
                "Required when a mel pack is in --train_file: those packs store paths, not "
                "audio, and the waveform is read here in the dataloader workers."
            )
        },
    )
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
        self.mel = WhisperMel()
        self.mel_projector = MelProjector(config.hidden_size)
        self.loss = LigerFusedLinearCrossEntropyLoss(reduction="sum")

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        labels=None,
        mel_waveform=None,
        mel_frame_index=None,
        mel_segment_id=None,
        mel_num_segments=0,
        num_items_in_batch=None,
        **kwargs,
    ):
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if mel_waveform is not None and mel_waveform.numel():
            # fp32 throughout: an STFT in bf16 loses the quiet end of the spectrum, and
            # whisper's (max - 8) floor makes that loss utterance-wide
            with torch.autocast(device_type=inputs_embeds.device.type, enabled=False):
                mel = self.mel(
                    mel_waveform.float(), mel_frame_index, mel_segment_id, int(mel_num_segments)
                )
            features = self.mel_projector(mel)
            slots = input_ids == self.config.mel_token_id
            if int(slots.sum()) != features.shape[0]:
                raise RuntimeError(
                    f'{int(slots.sum())} <|mel|> placeholders but {features.shape[0]} mel positions '
                    '— the pack and the mel front end disagree on the frame rate'
                )
            # out-of-place: an in-place index_put_ into the embedding output does not
            # survive gradient checkpointing cleanly
            inputs_embeds = inputs_embeds.masked_scatter(
                slots.unsqueeze(-1), features.to(inputs_embeds.dtype)
            )
            mel_touch = None
        else:
            # no mel document in this micro-batch — keep the projector in the graph or
            # ddp_find_unused_parameters=false stalls the all-reduce (see zero_probe)
            mel_touch = self.mel_projector.zero_probe(inputs_embeds)

        super_out = self.model.forward(
            inputs_embeds = inputs_embeds,
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
            if num_items_in_batch is None:
                # masking the mel tokens out of the labels makes this a real count, not
                # just the token total the token-only trainers can assume
                num_items_in_batch = (labels != -100).sum()
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
            if mel_touch is not None:
                loss = loss + mel_touch.to(loss.dtype)
            return {'loss': loss}
        return super_out


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

    if not model_args.stt_tokens_file:
        logger.warning('no --stt_tokens_file: <|STT|> and the language tokens will be missing, '
                       'STT packs cannot be trained')
    tokenizer = build_speech_tokenizer(
        model_args.model_name_or_path,
        stt_tokens_file=model_args.stt_tokens_file,
        add_mel_tokens=True,
    )
    mel_token_id = tokenizer.convert_tokens_to_ids('<|mel|>')
    # every mel token is an input the model is fed, never an output it should emit:
    # the placeholders carry no identity, and <|mel_start|>/<|mel_end|> sit where the
    # length of an unseen audio span would have to be guessed
    mel_label_ids = np.array(tokenizer.convert_tokens_to_ids(MEL_TOKENS), dtype=np.int64)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    min_dtype = torch.finfo(torch_dtype).min
    sequence_length = data_args.block_size

    class DatasetFixed(torch.utils.data.Dataset):
        """One packed dataset; blocks carry audio only if the pack wrote it."""

        def __init__(self, local):
            self.dataset = StreamingDataset(local=local)

        def __getitem__(self, idx):
            # never mutate the row: the reader caches it (see unpack_block)
            return unpack_block(self.dataset[idx], sequence_length, data_args.audio_dir)

        def __len__(self):
            return len(self.dataset)

    class MixedDataset(torch.utils.data.Dataset):
        """Several packed datasets behind one flat index (see mixture_index).

        Repeats and slices are index maps -- nothing is copied, and the Trainer's
        sampler still shuffles one flat index space.
        """

        def __init__(self, specs):
            self.datasets = [DatasetFixed(path) for path, _ in specs]
            sizes = [len(ds) for ds in self.datasets]
            self.pairs = mixture_index(sizes, [w for _, w in specs])
            for i, ((path, weight), n) in enumerate(zip(specs, sizes)):
                taken = int((self.pairs[:, 0] == i).sum())
                print(f'{path}: {n} blocks, weight {weight} -> {taken} blocks')

        def __getitem__(self, idx):
            which, row = self.pairs[idx]
            return self.datasets[which][int(row)]

        def __len__(self):
            return len(self.pairs)

    model = Model.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation = 'kernels-community/vllm-flash-attn3',
        torch_dtype = model_args.torch_dtype,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False, pad_to_multiple_of=8)
    model.config.mel_token_id = mel_token_id
    print(model)
    # a submodule built in __init__ is initialised from from_pretrained's missing-keys
    # path; print it so a zeroed or nan projector shows up here and not as a 0.0 loss
    fc1 = model.mel_projector.fc1.weight
    print(f'mel_projector.fc1: std={float(fc1.std()):.5f} requires_grad={fc1.requires_grad}')

    specs = parse_train_files(data_args.train_file)
    if data_args.audio_dir and not os.path.isdir(data_args.audio_dir):
        raise SystemExit(f'--audio_dir {data_args.audio_dir} does not exist')
    dataset = MixedDataset(specs)
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
        labels[np.isin(labels, mel_label_ids)] = -100
        query_lens = np.concatenate(attention_mask)
        cumsum = [0] + np.cumsum(query_lens).tolist()
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

        # utterance order across the flattened batch has to match the order the <|mel|>
        # placeholders appear in, which is why the waveforms are laid out block by block
        waveforms = [w for b in batch for w in b.get('waveforms', ())]
        if waveforms:
            buffer, frame_index, segment_id = layout_segments(waveforms)
            out['mel_waveform'] = torch.from_numpy(buffer)
            out['mel_frame_index'] = torch.from_numpy(frame_index)
            out['mel_segment_id'] = torch.from_numpy(segment_id)
            out['mel_num_segments'] = len(waveforms)
        return out

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
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
