#!/usr/bin/env python3
"""
Async GRPO Training with Dedicated vLLM GPU
=============================================

Layout:
  GPU 0-6  → GRPO training (DDP, 7 processes)
  GPU 7    → vLLM inference (1 dedicated process)

Launch:
  torchrun --nproc_per_node=8 grpo_async_vllm.py

Design:
  - All 8 ranks participate in global collective ops (weight sync, generation handoff)
  - DDP gradient sync uses a training-only sub-group (ranks 0–6), so rank 7 is excluded
  - Weight sync: only ranks 0 and 7 participate (efficient 2-rank group broadcast)
  - Generation: rank 0 gathers all prompts → broadcasts to rank 7 → rank 7 generates
    → broadcasts completions back → each training rank slices its portion
  - Collective op ordering ensures no race between generation and weight loading

References:
  - TRL vllm_generation.py weight sync: https://github.com/huggingface/trl
"""

import argparse
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    get_cosine_schedule_with_warmup,
)

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Training
    learning_rate: float = 1e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2  # prompts per GPU per step
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.1
    warmup_ratio: float = 0.05

    # GRPO
    num_generations: int = 8  # completions per prompt (G)
    temperature: float = 0.9
    top_p: float = 1.0
    max_prompt_length: int = 512
    max_completion_length: int = 512
    beta: float = 0.04  # KL coefficient (0 = no ref model)
    epsilon: float = 0.2  # PPO clip range

    # vLLM
    vllm_gpu_memory_utilization: float = 0.85

    # Model
    attn_implementation: str = "flash_attention_2"  # "flash_attention_2" | "sdpa" | "eager"

    # Logging / saving
    output_dir: str = "./grpo_output"
    save_steps: int = 200
    logging_steps: int = 10


# ─── Weight Synchronization ──────────────────────────────────────────────────


class WeightSyncer:
    """
    Syncs DDP model weights from rank 0 to the vLLM rank after each optimizer step.

    Uses a dedicated 2-rank process group (ranks 0 and N-1) so that training
    ranks 1–6 are completely uninvolved in the weight transfer, saving bandwidth.

    Pre-caches parameter metadata (name, shape, dtype) and pre-allocates receive
    buffers on the vLLM rank to avoid per-step allocations.

    All weight sync is done per-parameter via NCCL broadcast within the 2-rank group.
    """

    def __init__(
        self,
        model: Optional[DDP],
        llm: Optional["LLM"],
        rank: int,
        world_size: int,
    ):
        self.model = model
        self.llm = llm
        self.rank = rank
        self.world_size = world_size
        self.vllm_rank = world_size - 1
        self.is_vllm = rank == self.vllm_rank
        # Only ranks 0 and vllm_rank participate in weight sync
        self.participates = rank == 0 or rank == self.vllm_rank
        self._param_meta: List[Tuple[str, torch.Size, torch.dtype]] = []
        self._cached_params: List[torch.Tensor] = []  # rank 0 only
        self._recv_buffers: Dict[str, torch.Tensor] = {}  # vllm_rank only

        # 2-rank group for weight transfer (avoids broadcasting to training ranks 1-6)
        self._sync_group = dist.new_group(ranks=[0, self.vllm_rank])
        self._init_meta()

    def _init_meta(self):
        """Broadcast parameter names/shapes/dtypes from rank 0 to rank 7."""
        if not self.is_vllm:
            meta = [
                (name, tuple(p.shape), p.dtype)
                for name, p in self.model.module.named_parameters()
            ]
        else:
            meta = None

        obj = [meta]
        dist.broadcast_object_list(obj, src=0)  # all 8 ranks participate
        self._param_meta = obj[0]

        if not self.is_vllm:
            # Cache parameter tensor references (avoids dict lookups during sync)
            self._cached_params = [
                p.data for _, p in self.model.module.named_parameters()
            ]
        else:
            # Pre-allocate GPU receive buffers on vLLM rank
            self._recv_buffers = {
                name: torch.empty(shape, dtype=dtype, device="cuda")
                for name, shape, dtype in self._param_meta
            }

    def _get_vllm_model(self):
        """Resolve the vLLM model runner (handles different vLLM versions)."""
        try:
            # vLLM v0.x / common path
            return self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        except AttributeError:
            try:
                # vLLM v1.x alternative path
                return self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
            except AttributeError:
                raise RuntimeError(
                    "Cannot resolve vLLM model runner. "
                    "Update WeightSyncer._get_vllm_model() for your vLLM version."
                )

    def sync(self):
        """
        Broadcast model weights from rank 0 to rank 7 (vLLM).

        Ranks 1–6 do NOT call this — only ranks 0 and 7 participate via _sync_group.
        Caller must ensure no vLLM generation is in-flight (use dist.barrier() first).
        """
        if not self.participates:
            return

        vllm_model = self._get_vllm_model() if self.is_vllm else None

        for i, (name, _shape, _dtype) in enumerate(self._param_meta):
            buf = self._recv_buffers[name] if self.is_vllm else self._cached_params[i]
            dist.broadcast(buf, src=0, group=self._sync_group)
            if self.is_vllm:
                vllm_model.load_weights([(name, buf)])

        if self.is_vllm:
            try:
                self.llm.reset_prefix_cache()
            except Exception:
                pass


# ─── Generation Coordinator ──────────────────────────────────────────────────


def generate_completions(
    local_prompt_ids: List[List[int]],
    config: GRPOConfig,
    rank: int,
    world_size: int,
    llm: Optional["LLM"],
    training_group: dist.ProcessGroup,
) -> Tuple[Optional[List[List[int]]], Optional[List[List[int]]]]:
    """
    Coordinate generation across all 8 ranks.

    Collective op sequence (all 8 ranks participate in steps marked [ALL]):
      1. Training ranks gather prompts on rank 0     (training sub-group)
      2. [ALL] Rank 0 broadcasts flat prompts+counts to all 8 ranks
      3. Rank 7 runs vLLM generation                 (local, no comms)
      4. [ALL] Rank 7 broadcasts completions to all 8 ranks
      5. Training ranks slice their local portion

    Returns:
      (prompt_ids_expanded, completion_ids) for training ranks,
      (None, None) for vLLM rank (rank 7).
    """
    vllm_rank = world_size - 1
    num_training = world_size - 1
    is_vllm = rank == vllm_rank

    # ── Step 1: Gather all prompts on rank 0 (training ranks only) ──
    if not is_vllm:
        gathered = [None] * num_training
        dist.all_gather_object(gathered, local_prompt_ids, group=training_group)
        if rank == 0:
            flat_prompts = [p for ps in gathered for p in ps]
            counts = [len(ps) for ps in gathered]
        else:
            flat_prompts, counts = None, None
    else:
        flat_prompts, counts = None, None

    # ── Step 2: Rank 0 broadcasts prompts to ALL 8 ranks (including rank 7) ──
    obj = [flat_prompts, counts]
    dist.broadcast_object_list(obj, src=0)
    flat_prompts, counts = obj

    # ── Step 3: vLLM rank generates ──
    if is_vllm:
        sampling_params = SamplingParams(
            n=config.num_generations,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_completion_length,
        )
        vllm_inputs = [{"prompt_token_ids": ids} for ids in flat_prompts]
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
        # Flatten: each prompt → num_generations completions
        all_completions = [list(out.token_ids) for o in outputs for out in o.outputs]
        all_prompts_exp = [ids for ids in flat_prompts for _ in range(config.num_generations)]
    else:
        all_completions, all_prompts_exp = None, None

    # ── Step 4: Rank 7 broadcasts completions to ALL 8 ranks ──
    result = [all_completions, all_prompts_exp]
    dist.broadcast_object_list(result, src=vllm_rank)
    all_completions, all_prompts_exp = result

    if is_vllm:
        return None, None

    # ── Step 5: Slice this rank's portion ──
    offset = sum(counts[:rank]) * config.num_generations
    n_local = counts[rank] * config.num_generations
    return (
        all_prompts_exp[offset: offset + n_local],
        all_completions[offset: offset + n_local],
    )


# ─── Data Utilities ───────────────────────────────────────────────────────────


def tokenize_prompts(texts: List[str], tokenizer, max_length: int) -> List[List[int]]:
    """Tokenize without padding; returns list of token id lists."""
    enc = tokenizer(texts, truncation=True, max_length=max_length, add_special_tokens=True)
    return enc["input_ids"]


def build_model_inputs(
    prompt_ids: List[List[int]],
    completion_ids: List[List[int]],
    tokenizer,
    max_total_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Concatenate prompt+completion, left-pad to same length.
    Returns (input_ids, attention_mask, completion_mask).
    completion_mask=1 only on completion token positions.
    """
    pad_id = tokenizer.pad_token_id
    seqs, cmasks = [], []
    for p_ids, c_ids in zip(prompt_ids, completion_ids):
        seq = (p_ids + c_ids)[:max_total_length]
        cm = ([0] * len(p_ids) + [1] * len(c_ids))[:max_total_length]
        seqs.append(seq)
        cmasks.append(cm)

    max_len = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids_t = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn_t = torch.zeros(B, max_len, dtype=torch.long, device=device)
    comp_t = torch.zeros(B, max_len, dtype=torch.long, device=device)

    for i, (s, cm) in enumerate(zip(seqs, cmasks)):
        start = max_len - len(s)  # left-pad
        input_ids_t[i, start:] = torch.tensor(s, dtype=torch.long, device=device)
        attn_t[i, start:] = 1
        comp_t[i, start:] = torch.tensor(cm, dtype=torch.long, device=device)

    return input_ids_t, attn_t, comp_t


# ─── GRPO Loss ────────────────────────────────────────────────────────────────


def compute_grpo_loss(
    model: DDP,
    ref_model: Optional[PreTrainedModel],
    input_ids: torch.Tensor,        # (B*G, L)
    attention_mask: torch.Tensor,   # (B*G, L)
    completion_mask: torch.Tensor,  # (B*G, L)
    rewards: torch.Tensor,          # (B*G,)
    config: GRPOConfig,
    old_log_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    GRPO loss = policy gradient with group-relative advantages + optional KL penalty.
    Supports PPO-clip (pass old_log_probs) or vanilla REINFORCE (old_log_probs=None).
    """
    G = config.num_generations
    B = rewards.shape[0] // G

    # Group-relative advantage normalization
    r = rewards.view(B, G)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-8)
    adv = adv.view(B * G)

    # Compute policy log probs
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B*G, L, V)
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = completion_mask[:, 1:].float()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)  # (B*G, L-1)
    seq_lp = (token_lp * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1)  # (B*G,)

    # Policy gradient loss (with optional PPO clip)
    if old_log_probs is not None:
        ratio = torch.exp(seq_lp - old_log_probs)
        pg_loss = -torch.min(
            ratio * adv,
            torch.clamp(ratio, 1 - config.epsilon, 1 + config.epsilon) * adv,
        ).mean()
    else:
        pg_loss = -(seq_lp * adv).mean()

    # KL penalty vs frozen reference model
    kl_loss = torch.tensor(0.0, device=input_ids.device)
    if ref_model is not None and config.beta > 0:
        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits
        ref_lp = F.log_softmax(ref_logits[:, :-1], dim=-1)
        ref_token_lp = ref_lp.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
        # Schulman's unbiased estimator: exp(log_p - log_ref) - (log_p - log_ref) - 1
        log_ratio = token_lp - ref_token_lp
        kl_tok = (torch.exp(log_ratio) - log_ratio - 1) * shift_mask
        kl_loss = kl_tok.sum(-1).mean()

    loss = pg_loss + config.beta * kl_loss

    return loss, {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "kl": kl_loss.item(),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "adv_mean": adv.mean().item(),
    }


# ─── vLLM Init ────────────────────────────────────────────────────────────────


def init_vllm(config: GRPOConfig) -> "LLM":
    """
    Initialize vLLM on the dedicated GPU.

    Temporarily hides distributed env vars so vLLM doesn't try to init
    its own NCCL group on top of the existing torch.distributed world.
    After init, env vars are restored for subsequent NCCL communication.
    """
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM not installed. Run: pip install vllm")

    dist_keys = [
        "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
    ]
    saved = {k: os.environ.pop(k, None) for k in dist_keys}
    try:
        llm = LLM(
            model=config.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            max_model_len=config.max_prompt_length + config.max_completion_length,
            dtype="bfloat16",
            enforce_eager=False,
        )
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    return llm


# ─── Main Training Loop ───────────────────────────────────────────────────────


def run(
    config: GRPOConfig,
    reward_fn: Callable[[List[str], List[str]], List[float]],
    train_prompts: List[str],
):
    """
    Main entry point — called by all 8 ranks via torchrun.

    rank 7   → initializes vLLM, participates in collective ops, skips optimizer
    ranks 0–6 → DDP training, reward computation, GRPO loss
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(config.__dict__.get("seed", 42) + rank)

    vllm_rank = world_size - 1
    is_vllm = rank == vllm_rank
    num_training = world_size - 1  # 7 training GPUs
    device = torch.device(f"cuda:{local_rank}")

    # Sub-group for DDP gradient sync (excludes vLLM rank)
    training_group = dist.new_group(ranks=list(range(num_training)))

    # ── Initialize model or vLLM ──
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not is_vllm:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=config.attn_implementation,
        ).to(device)
        # Use training-only process group so DDP never touches rank 7
        model = DDP(model, device_ids=[local_rank], process_group=training_group)

        ref_model = None
        if config.beta > 0:
            ref_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation=config.attn_implementation,
            ).to(device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)

        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        llm = None

        # Shard dataset across 7 training ranks
        shard = len(train_prompts) // num_training
        local_prompts = train_prompts[rank * shard: (rank + 1) * shard]
    else:
        model, ref_model, optimizer, local_prompts = None, None, None, []
        llm = init_vllm(config)

    # ── Build WeightSyncer (all 8 ranks, broadcasts param metadata) ──
    dist.barrier()
    syncer = WeightSyncer(model, llm, rank, world_size)

    # ── Determine total steps (all ranks must agree) ──
    if rank == 0:
        steps_per_epoch = max(1, len(local_prompts) // config.per_device_train_batch_size)
        total_steps = config.num_train_epochs * steps_per_epoch
    else:
        total_steps, steps_per_epoch = 0, 0
    obj = [total_steps, steps_per_epoch]
    dist.broadcast_object_list(obj, src=0)
    total_steps, steps_per_epoch = obj

    if not is_vllm:
        warmup = max(1, int(total_steps * config.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        print(f"[GRPO] total_steps={total_steps}  training_gpus={num_training}  "
              f"vllm_gpu={vllm_rank}  G={config.num_generations}")

    # ── Initial weight sync so vLLM matches the training model ──
    syncer.sync()
    dist.barrier()

    # ── Training loop (all 8 ranks iterate the same count) ──
    for step in range(total_steps):

        # Select batch for training ranks
        if not is_vllm:
            ep_step = step % steps_per_epoch
            bs = config.per_device_train_batch_size
            batch_texts = local_prompts[ep_step * bs: (ep_step + 1) * bs]
            prompt_ids = tokenize_prompts(batch_texts, tokenizer, config.max_prompt_length)
        else:
            prompt_ids = []

        # ── Generation (all 8 ranks participate in collective ops) ──
        prompt_ids_exp, completion_ids = generate_completions(
            local_prompt_ids=prompt_ids,
            config=config,
            rank=rank,
            world_size=world_size,
            llm=llm,
            training_group=training_group,
        )

        # ── Training (ranks 0–6 only) ──
        if not is_vllm:
            # Decode for reward function
            prompts_str = tokenizer.batch_decode(prompt_ids_exp, skip_special_tokens=True)
            completions_str = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            rewards = torch.tensor(
                reward_fn(prompts_str, completions_str),
                dtype=torch.float32,
                device=device,
            )

            max_total = config.max_prompt_length + config.max_completion_length
            input_ids, attn_mask, comp_mask = build_model_inputs(
                prompt_ids_exp, completion_ids, tokenizer, max_total, device
            )

            model.train()
            optimizer.zero_grad()

            G = config.num_generations
            chunk = max(1, len(input_ids) // config.gradient_accumulation_steps)
            metrics_acc: Dict[str, float] = {}

            for acc in range(config.gradient_accumulation_steps):
                sl = slice(acc * chunk, (acc + 1) * chunk)
                # no_sync() avoids gradient all-reduce on intermediate accumulation steps
                ctx = model.no_sync() if acc < config.gradient_accumulation_steps - 1 else nullcontext()
                with ctx:
                    loss, m = compute_grpo_loss(
                        model, ref_model,
                        input_ids[sl], attn_mask[sl], comp_mask[sl],
                        rewards[sl], config,
                    )
                    (loss / config.gradient_accumulation_steps).backward()
                for k, v in m.items():
                    metrics_acc[k] = metrics_acc.get(k, 0.0) + v / config.gradient_accumulation_steps

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if rank == 0 and (step + 1) % config.logging_steps == 0:
                log = " | ".join(f"{k}: {v:.4f}" for k, v in metrics_acc.items())
                print(f"Step {step + 1}/{total_steps} | {log}")

        # ── Weight sync to vLLM after optimizer step (ranks 0 and 7 only) ──
        # barrier ensures vLLM rank is not in active generate() when we load weights
        dist.barrier()
        syncer.sync()

        # ── Save checkpoint (rank 0) ──
        if not is_vllm and rank == 0 and (step + 1) % config.save_steps == 0:
            ckpt = f"{config.output_dir}/checkpoint-{step + 1}"
            model.module.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"Saved checkpoint → {ckpt}")

    # ── Final save ──
    if not is_vllm and rank == 0:
        model.module.save_pretrained(f"{config.output_dir}/final")
        tokenizer.save_pretrained(f"{config.output_dir}/final")
        print("Training complete.")

    dist.destroy_process_group()


# ─── Example ─────────────────────────────────────────────────────────────────


def example_reward_fn(prompts: List[str], completions: List[str]) -> List[float]:
    """
    Placeholder reward function. Replace with your actual reward logic.
    Example: reward = completion length normalized to [0, 1].
    """
    return [min(len(c.split()) / 50.0, 1.0) for c in completions]


def parse_args() -> GRPOConfig:
    defaults = GRPOConfig()
    p = argparse.ArgumentParser(
        description="Async GRPO training with dedicated vLLM GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model_name", type=str, default=defaults.model_name)
    p.add_argument(
        "--attn_implementation", type=str, default=defaults.attn_implementation,
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention backend for the training model",
    )

    # Training
    p.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    p.add_argument("--num_train_epochs", type=int, default=defaults.num_train_epochs)
    p.add_argument("--per_device_train_batch_size", type=int, default=defaults.per_device_train_batch_size)
    p.add_argument("--gradient_accumulation_steps", type=int, default=defaults.gradient_accumulation_steps)
    p.add_argument("--max_grad_norm", type=float, default=defaults.max_grad_norm)
    p.add_argument("--warmup_ratio", type=float, default=defaults.warmup_ratio)

    # GRPO
    p.add_argument("--num_generations", type=int, default=defaults.num_generations,
                   help="Completions per prompt (G)")
    p.add_argument("--temperature", type=float, default=defaults.temperature)
    p.add_argument("--top_p", type=float, default=defaults.top_p)
    p.add_argument("--max_prompt_length", type=int, default=defaults.max_prompt_length)
    p.add_argument("--max_completion_length", type=int, default=defaults.max_completion_length)
    p.add_argument("--beta", type=float, default=defaults.beta,
                   help="KL penalty coefficient (0 = no reference model)")
    p.add_argument("--epsilon", type=float, default=defaults.epsilon,
                   help="PPO clip range")

    # vLLM
    p.add_argument("--vllm_gpu_memory_utilization", type=float,
                   default=defaults.vllm_gpu_memory_utilization)

    # Logging / saving
    p.add_argument("--output_dir", type=str, default=defaults.output_dir)
    p.add_argument("--save_steps", type=int, default=defaults.save_steps)
    p.add_argument("--logging_steps", type=int, default=defaults.logging_steps)

    args = p.parse_args()
    return GRPOConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()

    train_prompts = [
        "Solve step by step: What is 15 × 24?",
        "Explain the difference between supervised and unsupervised learning.",
        "Write a Python function to check if a number is prime.",
        "What are the key principles of SOLID design?",
        "Describe the water cycle in detail.",
    ] * 500  # pad to enough steps

    run(
        config=config,
        reward_fn=example_reward_fn,
        train_prompts=train_prompts,
    )
