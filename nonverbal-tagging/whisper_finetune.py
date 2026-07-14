#!/usr/bin/env python3
"""Fine-tune Whisper to emit inline non-verbal tags ([Laughter], [Cough], ...) — plan.md step 2.

Consumes jsonl manifests (one object per line):

    {"audio_path": "/data/x.mp3", "text": "Ada ke tau? [Laughter] Tak adalah...", "language": "ms"}

- `text` is the target transcript WITH inline tags (nv_text for positives; plain transcript for
  negatives and CLAP-rejected hard negatives — same format, just no tags).
- `language`: whisper language code (ms/en/ta/zh...). Per-sample decoder prefix is built manually
  so one run can mix languages.
- Tags are trained as plain text (multi-token BPE) — no vocab surgery in v1.

Eval gates (computed on --manifest-eval with predict_with_generate):
  1. tag_f1 — multiset precision/recall/F1 over [Tag] tokens per utterance
  2. cer_notags — CER with tags stripped: must not regress vs the base model

Examples:
  # LoRA on a 24 GB card
  python whisper_finetune.py --manifest-train train.jsonl --manifest-eval eval.jsonl \
      --output-dir out-lora --lora --batch-size 16 --grad-accum 2 --lr 1e-4

  # full fine-tune (H100)
  python whisper_finetune.py --manifest-train train.jsonl --manifest-eval eval.jsonl \
      --output-dir out-full --batch-size 32 --lr 1e-5 --freeze-encoder
"""
import argparse
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

SR = 16000
TAG_RE = re.compile(r"\[[A-Z][A-Za-z-]*\]")  # [Laughter], [Cough], ...


# ---------------------------------------------------------------- data

class ManifestDataset(Dataset):
    """jsonl manifest -> (input_features, labels). Audio decoded lazily in workers."""

    def __init__(self, manifest, processor, max_dur=30.0):
        self.rows = [json.loads(l) for l in open(manifest)]
        self.processor = processor
        self.max_samples = int(max_dur * SR)
        tok = processor.tokenizer
        self.transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
        self.notimestamps = tok.convert_tokens_to_ids("<|notimestamps|>")
        self.eot = tok.eos_token_id
        self._lang_cache = {}

    def _lang_id(self, lang):
        if lang not in self._lang_cache:
            tid = self.processor.tokenizer.convert_tokens_to_ids(f"<|{lang}|>")
            unk = self.processor.tokenizer.unk_token_id
            if tid is None or tid == unk:
                tid = self.processor.tokenizer.convert_tokens_to_ids("<|en|>")
            self._lang_cache[lang] = tid
        return self._lang_cache[lang]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        y, _ = librosa.load(r["audio_path"], sr=SR, mono=True)
        y = y[: self.max_samples]
        feats = self.processor.feature_extractor(
            y, sampling_rate=SR, return_tensors="np"
        ).input_features[0]

        text_ids = self.processor.tokenizer(
            r["text"].strip(), add_special_tokens=False
        ).input_ids
        # no <|startoftranscript|> here: the model prepends it (decoder_start_token_id)
        # when shifting labels right to build decoder inputs.
        labels = [self._lang_id(r.get("language", "en")),
                  self.transcribe, self.notimestamps] + text_ids + [self.eot]
        return {"input_features": feats, "labels": labels}


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, batch):
        feats = torch.tensor(np.stack([b["input_features"] for b in batch]))
        maxlen = max(len(b["labels"]) for b in batch)
        labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            labels[i, : len(b["labels"])] = torch.tensor(b["labels"])
        # trainer builds decoder_input_ids by shifting labels right
        return {"input_features": feats, "labels": labels}


# ---------------------------------------------------------------- metrics

def cer(ref, hyp):
    """character error rate via levenshtein (no jiwer dependency)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def build_compute_metrics(processor):
    def compute_metrics(pred):
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        tp = fp = fn = 0
        cers = []
        for h, r in zip(hyps, refs):
            htags, rtags = TAG_RE.findall(h), TAG_RE.findall(r)
            for t in set(htags) | set(rtags):
                nh, nr = htags.count(t), rtags.count(t)
                tp += min(nh, nr)
                fp += max(0, nh - nr)
                fn += max(0, nr - nh)
            h_clean = re.sub(r"\s+", " ", TAG_RE.sub(" ", h)).strip().lower()
            r_clean = re.sub(r"\s+", " ", TAG_RE.sub(" ", r)).strip().lower()
            cers.append(cer(r_clean, h_clean))

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return {
            "tag_precision": round(prec, 4),
            "tag_recall": round(rec, 4),
            "tag_f1": round(f1, 4),
            "cer_notags": round(float(np.mean(cers)), 4),
        }

    return compute_metrics


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-train", required=True)
    ap.add_argument("--manifest-eval", required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lora", action="store_true", help="LoRA via peft (24 GB-card path)")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-4 lora, 1e-5 full")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--gen-max-length", type=int, default=440)
    args = ap.parse_args()

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16
    )
    # per-sample prefixes live in the labels; disable whisper's static forcing
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    if args.freeze_encoder and not args.lora:
        model.model.encoder.requires_grad_(False)

    if args.lora:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    train_ds = ManifestDataset(args.manifest_train, processor)
    eval_ds = ManifestDataset(args.manifest_eval, processor)
    print(f"train: {len(train_ds)}, eval: {len(eval_ds)}")

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr or (1e-4 if args.lora else 1e-5),
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=not args.lora,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=3,
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=args.gen_max_length,
        load_best_model_at_end=True,
        metric_for_best_model="tag_f1",
        greater_is_better=True,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=Collator(processor.tokenizer.pad_token_id),
        compute_metrics=build_compute_metrics(processor),
    )

    print("=== baseline eval (pre-training) ===")
    print(trainer.evaluate())

    trainer.train()
    trainer.save_model(f"{args.output_dir}/best")
    processor.save_pretrained(f"{args.output_dir}/best")
    print("=== final eval ===")
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
