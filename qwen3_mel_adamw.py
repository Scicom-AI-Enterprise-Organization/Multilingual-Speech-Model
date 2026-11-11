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

`--mel false` trains tasks 1 and 2 only: the model is then a plain Qwen3ForCausalLM (no
WhisperMel, no projector, nothing extra in the checkpoint) and a mel pack in the data is
an error. The `<|mel*|>` tokens stay in the vocab, so every id matches the three-task runs.

Optimizer is a launch flag (`--optimizer`, see speech_optimizers.py). The three-task
FLEURS sweep at the published batch size ranked `soap --matrix_lr 1e-3` first on mean dev
loss and diverged AdamW at 1e-3, so SOAP is the default here — the older AdamW-won result
was a token-only, one-epoch ablation (README_V1).

The loss is a launch flag too (`--loss_impl liger | fused`): both are fused linear
cross-entropies that never hold the [tokens x vocab] logits; `fused` is fused_ce.py from
small-ablation/cross-entropy and computes the logits themselves in fp32.
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
from fused_ce import fused_linear_cross_entropy
from speech_optimizers import build_optimizer

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
    mel: bool = field(
        default=True,
        metadata={"help": "Build the raw-mel input path (WhisperMel + MelProjector). false = "
                          "TTS + STT on audio tokens only: the model and its checkpoints are a "
                          "plain Qwen3ForCausalLM and a mel pack in the data is an error. The "
                          "<|mel*|> tokens stay in the vocab either way, so no id moves."},
    )
    loss_impl: str = field(
        default='liger',
        metadata={"help": "liger | fused. Both never hold the [tokens x vocab] logits. fused is "
                          "fused_ce.py (small-ablation/cross-entropy): fp32 softmax statistics and "
                          "gradient sums, and the logits in --fused_compute_dtype.",
                  "choices": ["liger", "fused"]},
    )
    fused_compute_dtype: str = field(
        default='float32',
        metadata={"help": "fused: dtype of the output layer's products. float32 | bfloat16 | auto "
                          "(the wider of the hidden state's and lm_head's: bf16 on a bf16 model, "
                          "i.e. bf16 logits, the same rounding liger has).",
                  "choices": ["float32", "bfloat16", "auto"]},
    )
    fused_precision: str = field(
        default='tf32',
        metadata={"help": "fused, float32 products: tf32 (TF32 tensor cores) | ieee (exact fp32). "
                          "A bf16 input fits TF32's mantissa, so on a bf16 model tf32 loses only "
                          "in the tensor-core accumulation: loss error 2.3e-7 vs ieee's 7.5e-11 at "
                          "the 1.7B shape on H20, both far under bf16 logits' 3.4e-6, and ieee "
                          "costs 2.3x the time. On fp32 weights tf32 also rounds the inputs.",
                  "choices": ["tf32", "ieee"]},
    )
    fused_strategy: str = field(
        default='precompute',
        metadata={"help": "fused: precompute (3 products; gradients made in the forward and "
                          "held until backward) | recompute (4 products, least memory held).",
                  "choices": ["precompute", "recompute"]},
    )
    fused_block_mb: int = field(
        default=512,
        metadata={"help": "fused: size of the one fp32 logits block held at a time, MB."},
    )
    fused_filter_eps: float = field(
        default=0.0,
        metadata={"help": "fused: skip the gradient of vocabulary blocks whose probabilities are "
                          "all below this and that hold no target (CCE's filtering). 0 = exact."},
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
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Dev pack to validate on, or several named ones: "
                          "'tts=dirA,stt=dirB,mel=dirC' reports eval_tts_loss, eval_stt_loss "
                          "and eval_mel_loss separately. One model trains on the mixture, but "
                          "each task is scored on its own dev split -- a mixture-wide loss "
                          "hides an arm that has stopped learning."})
    max_eval_blocks: int = field(
        default=256,
        metadata={"help": "Evaluate on an evenly-strided subset this many blocks wide "
                          "(0 = all of it). A full dev pass costs real training time."})
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


@dataclass
class OptimizerArguments:
    """Which optimizer to train with. Defaults are the FLEURS three-task sweep's winner."""

    optimizer: str = field(
        default='soap',
        metadata={"help": "adamw | muon | shampoo | soap | lion | ademamix. The hybrids run the "
                          "2D hidden weights on the named optimizer at --matrix_lr and everything "
                          "else (embeddings, lm_head, norms) on AdamW at --learning_rate."},
    )
    matrix_lr: Optional[float] = field(
        default=1e-3,
        metadata={"help": "LR for the 2D-hidden-weight sub-optimizer; required for muon/shampoo/soap. "
                          "Sweep order: soap 1e-3 > soap 5e-4 > muon 1e-2 > muon 5e-3."},
    )


class Model(Qwen3ForCausalLM):
    """TTS + STT on audio tokens. `MelModel` adds the raw-mel input path.

    Kept free of mel modules so a token-only run's checkpoint is a plain
    Qwen3ForCausalLM and its optimizer carries no projector that never trains.
    """

    # set by main() after from_pretrained; class defaults keep a bare from_pretrained usable
    loss_impl = 'liger'
    fused_kwargs = {}

    def __init__(self, config):
        super().__init__(config)
        self.loss = LigerFusedLinearCrossEntropyLoss(reduction="sum")

    def embed(self, input_ids, mel_waveform=None, mel_frame_index=None, mel_segment_id=None,
              mel_num_segments=0):
        """Input embeddings, and a 0-valued term the loss must carry (None when not needed)."""
        if mel_waveform is not None and mel_waveform.numel():
            raise RuntimeError('a raw-mel block reached a model built with --mel false: drop the '
                               'mel packs from --train_file / --validation_file, or use --mel true')
        return self.get_input_embeddings()(input_ids), None

    def summed_loss(self, hidden, labels):
        """Next-token cross-entropy summed over the labelled positions, logits never held."""
        if self.loss_impl == 'fused':
            return fused_linear_cross_entropy(
                hidden, self.lm_head.weight, labels, shift=True, reduction='sum',
                **self.fused_kwargs,
            )
        hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        labels = labels[..., 1:].contiguous().reshape(-1)
        return self.loss(self.lm_head.weight, hidden, labels)

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
        inputs_embeds, probe = self.embed(
            input_ids, mel_waveform, mel_frame_index, mel_segment_id, mel_num_segments)

        super_out = self.model.forward(
            inputs_embeds = inputs_embeds,
            position_ids = position_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
            **kwargs,
        )
        if labels is None:
            return super_out
        loss = self.summed_loss(super_out.last_hidden_state, labels)
        if num_items_in_batch is None:
            # masking the mel tokens out of the labels makes this a real count, not
            # just the token total the token-only trainers can assume
            num_items_in_batch = (labels[..., 1:] != -100).sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
        if probe is not None:
            loss = loss + probe.to(loss.dtype)
        return {'loss': loss}


class MelModel(Model):
    """Model + the raw log-mel front end: `<|mel|>` embeddings come from the audio."""

    def __init__(self, config):
        super().__init__(config)
        self.mel = WhisperMel()
        self.mel_projector = MelProjector(config.hidden_size)

    def embed(self, input_ids, mel_waveform=None, mel_frame_index=None, mel_segment_id=None,
              mel_num_segments=0):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if mel_waveform is None or not mel_waveform.numel():
            # no mel document in this micro-batch — keep the projector in the graph or
            # ddp_find_unused_parameters=false stalls the all-reduce (see zero_probe)
            return inputs_embeds, self.mel_projector.zero_probe(inputs_embeds)

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
        return inputs_embeds, None


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OptimizerArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, optim_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, optim_args, training_args = parser.parse_args_into_dataclasses()

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

        def __init__(self, local, indices=None):
            self.local = local
            self.dataset = StreamingDataset(local=local)
            # evenly-strided view, for capping the dev pass
            self.indices = indices

        def __getitem__(self, idx):
            if self.indices is not None:
                idx = int(self.indices[idx])
            row = self.dataset[idx]
            samples = row.get('audio_samples')
            if not model_args.mel and samples is not None and len(samples):
                # caught here, before unpack_block asks for an --audio_dir this run
                # deliberately does not have
                raise ValueError(f'{self.local} is a raw-mel pack but --mel false: '
                                 'drop it from --train_file / --validation_file')
            # never mutate the row: the reader caches it (see unpack_block)
            return unpack_block(row, sequence_length, data_args.audio_dir)

        def __len__(self):
            return len(self.dataset) if self.indices is None else len(self.indices)

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

    model = (MelModel if model_args.mel else Model).from_pretrained(
        model_args.model_name_or_path,
        attn_implementation = 'kernels-community/vllm-flash-attn3',
        torch_dtype = model_args.torch_dtype,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False, pad_to_multiple_of=8)
    model.config.mel_token_id = mel_token_id
    model.loss_impl = model_args.loss_impl
    if model_args.loss_impl == 'fused':
        model.fused_kwargs = dict(
            compute_dtype=(None if model_args.fused_compute_dtype == 'auto'
                           else getattr(torch, model_args.fused_compute_dtype)),
            fp32_precision=model_args.fused_precision,
            strategy=model_args.fused_strategy,
            block_mb=model_args.fused_block_mb,
            filter_eps=model_args.fused_filter_eps or None,
        )
    print(model)
    print(f'loss: {model.loss_impl} {model.fused_kwargs}')
    if model_args.mel:
        # a submodule built in __init__ is initialised from from_pretrained's missing-keys
        # path; print it so a zeroed or nan projector shows up here and not as a 0.0 loss
        fc1 = model.mel_projector.fc1.weight
        print(f'mel_projector.fc1: std={float(fc1.std()):.5f} requires_grad={fc1.requires_grad}')

    specs = parse_train_files(data_args.train_file)
    if data_args.audio_dir and not os.path.isdir(data_args.audio_dir):
        raise SystemExit(f'--audio_dir {data_args.audio_dir} does not exist')
    dataset = MixedDataset(specs)
    print('dataset', len(dataset), dataset[0]['attention_mask'].shape)

    def capped(path):
        """A dev pack, cut to --max_eval_blocks by striding rather than truncating.

        Packs are written worker by worker, so the first N blocks are one slice of the
        corpus rather than a sample of it.
        """
        ds = DatasetFixed(path)
        total = len(ds)
        cap = data_args.max_eval_blocks
        if cap and total > cap:
            ds.indices = np.linspace(0, total, cap, endpoint=False).astype(np.int64)
        return ds, total

    eval_dataset = None
    if data_args.validation_file:
        named = [part.split('=', 1) for part in data_args.validation_file.split(',') if part.strip()]
        if len(named) and all(len(part) == 2 for part in named):
            eval_dataset = {}
            for name, path in named:
                eval_dataset[name.strip()], total = capped(path.strip())
                print(f'eval dataset [{name.strip()}]', len(eval_dataset[name.strip()]),
                      f'of {total} blocks')
        else:
            eval_dataset, total = capped(data_args.validation_file)
            print('eval dataset', len(eval_dataset), f'of {total} blocks')

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

    # Trainer would build its own AdamW; pass ours and leave the scheduler to Trainer so
    # --lr_scheduler_type warmup_stable_decay + --lr_scheduler_kwargs stay authoritative
    optimizer = build_optimizer(
        model, optim_args.optimizer,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        matrix_lr=optim_args.matrix_lr,
    )
    print(optimizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
        optimizers=(optimizer, None),
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
        print('final param group LRs:', [g['lr'] for g in optimizer.param_groups])


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
