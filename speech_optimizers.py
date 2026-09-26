"""Optimizers shared by the sweep harness and the real training runs.

Extracted from `qwen3_optimizer_search.py` so a production trainer can use the sweep's
winner without importing a search script. Behaviour is unchanged — the sweep imports
these same definitions.

The FLEURS three-task ablation (README) ranked mean dev loss across TTS tokens, STT tokens
and STT raw mel over 200 steps at the published batch size:

    soap  matrix_lr 1e-3   5.4178   <- winner
    soap  matrix_lr 5e-4   5.4360
    muon  matrix_lr 1e-2   5.5534
    muon  matrix_lr 5e-3   5.6174
    soap  matrix_lr 1e-4   5.6785
    soap  matrix_lr 3e-3   6.2485
    shampoo 1e-2 / 3e-3    6.3104 / 6.3210
    adamw 5e-4             8.7507
    adamw 1e-3             diverged (grad_norm 272 -> 45,610 at peak LR)

so `--optimizer soap --matrix_lr 1e-3` is the default worth reaching for, not AdamW.
"""

import torch


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


def build_optimizer(model, name, lr, weight_decay, matrix_lr=None,
                    preconditioning_compute_steps=None):
    """`name` in adamw | muon | shampoo | soap | lion | ademamix.

    The hybrids (muon/shampoo/soap) need `matrix_lr`: the 2D hidden weights run on the
    matrix optimizer at that LR while everything else stays on AdamW at `lr`.

    `preconditioning_compute_steps` only affects shampoo. The sweep refreshed every 10
    steps because the library default (1000) would never refresh inside a 100-step run;
    a real run should leave it at the default.
    """
    name = name.lower()
    hybrid = name in ('muon', 'shampoo', 'soap')
    if hybrid and matrix_lr is None:
        raise ValueError(f'matrix_lr is required for optimizer={name}')

    if name == 'adamw':
        return torch.optim.AdamW(
            decay_param_groups(model.named_parameters(), lr, weight_decay),
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
            kwargs = {}
            if preconditioning_compute_steps is not None:
                kwargs = dict(start_preconditioning_step=10,
                              preconditioning_compute_steps=preconditioning_compute_steps,
                              statistics_compute_steps=1)
            return ScalableShampoo(
                params, lr=mlr, betas=(0.9, 0.999), weight_decay=decay,
                decoupled_weight_decay=True, **kwargs,
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
            decay_param_groups(model.named_parameters(), lr, weight_decay),
            lr=lr, betas=(0.9, 0.99), weight_decouple=True,
        )
    elif name == 'ademamix':
        from pytorch_optimizer import AdEMAMix
        return AdEMAMix(
            decay_param_groups(model.named_parameters(), lr, weight_decay),
            lr=lr, betas=(0.9, 0.999, 0.9999), alpha=5.0, weight_decouple=True,
        )
    else:
        raise ValueError(f'unknown optimizer {name!r}')

    return HybridOptimizer(
        model.named_parameters(),
        matrix_factory=factory,
        lr=lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
    )
