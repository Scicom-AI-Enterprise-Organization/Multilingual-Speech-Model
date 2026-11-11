"""Fused linear cross-entropy that never builds the logits and supports fp32 end to end.

Like Cut Cross-Entropy (Wijmans et al., 2025), the output layer and the loss are one autograd function: the [tokens x vocabulary] logits are
never held, only one block of them at a time. Unlike CCE's kernels, which compute in bf16/fp16, every product here can run in fp32
(`compute_dtype=torch.float32`), exactly ("ieee") or on TF32 tensor cores ("tf32"), and every softmax statistic is fp32 whatever the inputs.

    loss = fused_linear_cross_entropy(hidden, lm_head.weight, labels, shift=True)      # hidden [..., d], labels [...]

Forward, per block of tokens: the vocabulary is swept one block at a time with an online log-sum-exp (running maximum and sum, fp32), the
block of logits computed into one reused buffer and exponentiated in place; each token's target logit is read from the very block that
holds it, so it is rounded exactly as the sum was. Kept for backward: the inputs and one fp32 number per token.

Backward, per block of vocabulary (outer) and of tokens (inner): the block of logits is computed again into the same buffer, turned into
softmax probabilities in place, the target's 1 subtracted and the row's weight in the loss applied, then added into the hidden states'
gradient (addmm into an fp32 buffer of [tokens, d]) and the weight's (addmm into its rows, written once per vocabulary block). With
`filter_eps`, a block whose largest probability is below it and that holds no target skips both products, as CCE's gradient filtering does;
None (the default) keeps the gradient exact.

Two strategies. "precompute" (the default when gradients are wanted) does three products instead of four: a block of tokens holds its
logits over the whole vocabulary, so its log-sum-exp is exact at once and its gradients are made right there in the forward; the backward only
scales them in place. "recompute" is the online, four-product path above, the least memory between forward and backward.

Memory beyond the inputs: one logits block (`block_mb`, default 512 MB), the fp32 gradient of the hidden states, and the weight's gradient,
which autograd needs anyway. Only tokens whose label is not `ignore_index` are scored.

Vendored unchanged from Scicom-AI-Enterprise-Organization/small-ablation, cross-entropy/fused_ce.py @ c5e00cb, where it is benchmarked
against fp64, plain logits, Liger and CCE (README there). Fix it upstream first; `test_fused_ce.py` here is the same test.
"""
import contextlib
import math

import torch

IGNORE_INDEX = -100


@contextlib.contextmanager
def _fp32_precision(mode):
    """TF32 ("tf32") or exact IEEE fp32 ("ieee") for the products inside, restored afterwards; None leaves the global setting alone."""
    if mode is None:
        yield
        return
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _sizes(n, v, d, block_mb, chunk_tokens, chunk_vocab):
    """Block shape: vocabulary blocks of chunk_vocab (default 8192, a multiple of 128), token blocks as many rows as fit block_mb in fp32."""
    cv = min(v, chunk_vocab or 8192)
    ct = chunk_tokens or max(128, int(block_mb * 2**20 // (4 * cv)) // 128 * 128)
    return min(n, ct), cv


class _FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, b, y, reduction, cdt, precision, ct, cv, filter_eps):
        n, v = h.shape[0], w.shape[0]
        acc = torch.float64 if cdt == torch.float64 else torch.float32           # softmax statistics: fp32, or fp64 for fp64 inputs
        lse = torch.empty(n, dtype=acc, device=h.device)
        z = torch.zeros(n, dtype=acc, device=h.device)
        hc = h if h.dtype == cdt else h.to(cdt)
        buf = torch.empty(min(n, ct), cv, dtype=cdt, device=h.device)
        with torch.autocast(device_type=h.device.type, enabled=False), _fp32_precision(precision):
            for i0 in range(0, n, ct):
                i1 = min(i0 + ct, n)
                hi, yi = hc[i0:i1], y[i0:i1]
                m = torch.full((i1 - i0,), -math.inf, dtype=acc, device=h.device)
                s = torch.zeros(i1 - i0, dtype=acc, device=h.device)
                for j0 in range(0, v, cv):
                    j1 = min(j0 + cv, v)
                    wj = w[j0:j1] if w.dtype == cdt else w[j0:j1].to(cdt)
                    lg = buf[: i1 - i0, : j1 - j0] if j1 - j0 == cv else torch.empty(i1 - i0, j1 - j0, dtype=cdt, device=h.device)
                    torch.mm(hi, wj.t(), out=lg)
                    if b is not None:
                        lg.add_(b[j0:j1].to(cdt))
                    lgf = lg if lg.dtype == acc else lg.to(acc)
                    hit = ((yi >= j0) & (yi < j1)).nonzero().squeeze(1)
                    if hit.numel():
                        z[i0 + hit] = lgf[hit, yi[hit] - j0]
                    m_new = torch.maximum(m, lgf.amax(1))
                    s.mul_(torch.exp(m - m_new)).add_(lgf.sub_(m_new[:, None]).exp_().sum(1))
                    m = m_new
                lse[i0:i1] = m + torch.log(s)
        loss = (lse - z).sum()
        if reduction == "mean":
            loss = loss / max(n, 1)
        ctx.save_for_backward(h, w, b if b is not None else torch.empty(0, device=h.device), y, lse)
        ctx.args = (reduction, cdt, precision, ct, cv, filter_eps, b is not None)
        return loss

    @staticmethod
    def backward(ctx, g):
        h, w, b, y, lse = ctx.saved_tensors
        reduction, cdt, precision, ct, cv, filter_eps, has_b = ctx.args
        b = b if has_b else None
        n, v = h.shape[0], w.shape[0]
        acc = torch.float64 if cdt == torch.float64 else torch.float32
        scale = g.to(acc) / max(n, 1) if reduction == "mean" else g.to(acc)
        need_h, need_w, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], has_b and ctx.needs_input_grad[2]
        hc = h if h.dtype == cdt else h.to(cdt)
        gh = torch.zeros(n, h.shape[1], dtype=acc, device=h.device) if need_h else None   # gradients always add up in fp32 (fp64 for fp64)
        gw = torch.zeros(w.shape, dtype=acc, device=h.device) if need_w else None
        gb = torch.zeros(v, dtype=acc, device=h.device) if need_b else None
        f32 = cdt == acc
        buf = torch.empty(min(n, ct), cv, dtype=cdt, device=h.device)
        with torch.autocast(device_type=h.device.type, enabled=False), _fp32_precision(precision):
            for j0 in range(0, v, cv):
                j1 = min(j0 + cv, v)
                wj = w[j0:j1] if w.dtype == cdt else w[j0:j1].to(cdt)
                for i0 in range(0, n, ct):
                    i1 = min(i0 + ct, n)
                    hi, yi = hc[i0:i1], y[i0:i1]
                    lg = buf[: i1 - i0, : j1 - j0] if j1 - j0 == cv else torch.empty(i1 - i0, j1 - j0, dtype=cdt, device=h.device)
                    torch.mm(hi, wj.t(), out=lg)
                    if b is not None:
                        lg.add_(b[j0:j1].to(cdt))
                    p = lg if lg.dtype == acc else lg.to(acc)
                    p.sub_(lse[i0:i1, None]).exp_()                               # softmax, from the stored log-sum-exp
                    hit = ((yi >= j0) & (yi < j1)).nonzero().squeeze(1)
                    if filter_eps is not None and hit.numel() == 0 and float(p.amax()) < filter_eps:
                        continue                                                 # nothing in this block moves a gradient by more than eps
                    if hit.numel():
                        p[hit, yi[hit] - j0] -= 1.0
                    p.mul_(scale)
                    pc = p if p.dtype == cdt else p.to(cdt)
                    if need_h:                                                   # in place when the products are fp32; else via one small fp32 add
                        gh[i0:i1].addmm_(pc, wj) if f32 else gh[i0:i1].add_(torch.mm(pc, wj).to(acc))
                    if need_w:
                        gw[j0:j1].addmm_(pc.t(), hi) if f32 else gw[j0:j1].add_(torch.mm(pc.t(), hi).to(acc))
                    if need_b:
                        gb[j0:j1] += p.sum(0)
        return (gh.to(h.dtype) if need_h else None, gw.to(w.dtype) if need_w else None, gb.to(b.dtype) if need_b else None,
                None, None, None, None, None, None, None)


class _FusedLinearCrossEntropyPrecompute(torch.autograd.Function):
    """Three products instead of four: each block of tokens holds its logits over the whole vocabulary (rows of [block, V], block_mb of them),
    so the log-sum-exp is exact at once and the gradients are computed right there in the forward, softmax in place; the backward only
    scales them by the incoming gradient, in place. With filter_eps, the gradient products of a vocabulary sub-block whose probabilities in
    this token block are all below it, and that holds no target, are skipped."""

    @staticmethod
    def forward(ctx, h, w, b, y, reduction, cdt, precision, ct, cv, filter_eps):
        n, v, d = h.shape[0], w.shape[0], h.shape[1]
        acc = torch.float64 if cdt == torch.float64 else torch.float32
        need_h, need_w, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], b is not None and ctx.needs_input_grad[2]
        gh = torch.empty(n, d, dtype=acc, device=h.device) if need_h else None
        gw = torch.zeros(v, d, dtype=acc, device=h.device) if need_w else None
        gb = torch.zeros(v, dtype=acc, device=h.device) if need_b else None
        hc = h if h.dtype == cdt else h.to(cdt)
        wc = w if w.dtype == cdt else w.to(cdt)
        scale = 1.0 / max(n, 1) if reduction == "mean" else 1.0
        total = torch.zeros((), dtype=acc, device=h.device)
        buf = torch.empty(min(n, ct), v, dtype=cdt, device=h.device)
        f32 = cdt == acc
        with torch.autocast(device_type=h.device.type, enabled=False), _fp32_precision(precision):
            for i0 in range(0, n, ct):
                i1 = min(i0 + ct, n)
                hi, yi = hc[i0:i1], y[i0:i1]
                lg = buf[: i1 - i0]
                torch.mm(hi, wc.t(), out=lg)
                if b is not None:
                    lg.add_(b.to(cdt))
                p = lg if f32 else lg.to(acc)
                rows = torch.arange(i1 - i0, device=h.device)
                m = p.amax(1)
                z = p[rows, yi]
                p.sub_(m[:, None]).exp_()
                ssum = p.sum(1)
                total += (m + torch.log(ssum) - z).sum()
                p.div_(ssum[:, None])                                            # softmax, in place
                if filter_eps is not None:                                       # judged on the probabilities themselves, before any scaling
                    big = torch.stack([p[:, j0:min(j0 + cv, v)].amax() for j0 in range(0, v, cv)]) >= filter_eps
                p[rows, yi] -= 1.0
                if scale != 1.0:
                    p.mul_(scale)
                pc = p if f32 else p.to(cdt)
                if filter_eps is None:
                    if need_h:
                        torch.mm(pc, wc, out=gh[i0:i1]) if f32 else gh[i0:i1].copy_(torch.mm(pc, wc))
                    if need_w:
                        gw.addmm_(pc.t(), hi) if f32 else gw.add_(torch.mm(pc.t(), hi).to(acc))
                    if need_b:
                        gb += p.sum(0)
                    continue
                if need_h:
                    gh[i0:i1].zero_()
                tgt = torch.zeros(v, dtype=torch.bool, device=h.device)
                tgt[yi] = True
                hold = torch.stack([tgt[j0:min(j0 + cv, v)].any() for j0 in range(0, v, cv)])
                for k in (big | hold).nonzero().squeeze(1).tolist():         # only the sub-blocks that move a gradient by eps or more
                    j0, j1 = k * cv, min(k * cv + cv, v)
                    pj = pc[:, j0:j1]
                    if need_h:
                        gh[i0:i1].addmm_(pj, wc[j0:j1]) if f32 else gh[i0:i1].add_(torch.mm(pj, wc[j0:j1]).to(acc))
                    if need_w:
                        gw[j0:j1].addmm_(pj.t(), hi) if f32 else gw[j0:j1].add_(torch.mm(pj.t(), hi).to(acc))
                    if need_b:
                        gb[j0:j1] += p[:, j0:j1].sum(0)
        del buf
        ctx.save_for_backward(*(t if t is not None else torch.empty(0, device=h.device) for t in (gh, gw, gb)))
        ctx.need = (need_h, need_w, need_b)
        ctx.dtypes = (h.dtype, w.dtype, b.dtype if b is not None else None)
        return total * scale

    @staticmethod
    def backward(ctx, g):
        gh, gw, gb = ctx.saved_tensors
        (need_h, need_w, need_b), (dh, dw, db) = ctx.need, ctx.dtypes
        out = []
        for need, t, dt in ((need_h, gh, dh), (need_w, gw, dw), (need_b, gb, db)):
            if not need:
                out.append(None)
                continue
            if not (g.numel() == 1 and float(g) == 1.0):
                t.mul_(g.to(t.dtype))                                            # in place: no second copy of the weight's gradient
            out.append(t if t.dtype == dt else t.to(dt))
        return (*out, None, None, None, None, None, None, None)


def fused_linear_cross_entropy(hidden, weight, labels, bias=None, *, shift=False, ignore_index=IGNORE_INDEX, reduction="mean",
                               compute_dtype=None, fp32_precision="ieee", block_mb=512, chunk_tokens=None, chunk_vocab=None, filter_eps=None,
                               strategy="precompute"):
    """cross_entropy(hidden @ weight.T + bias, labels) without the logits.

    hidden [..., d], labels [...]: any leading shape; shift=True scores token t+1 from position t (a causal language model).
    reduction: "mean" over the scored tokens, or "sum" (divide by the batch's token count yourself under gradient accumulation).
    compute_dtype: dtype of the products; default the wider of hidden's and weight's (fp32 weights give fp32). The softmax is always fp32.
    fp32_precision: "ieee" (exact fp32) or "tf32" (fp32 storage, TF32 tensor-core products) when the products run in fp32; None follows
    torch.backends.cuda.matmul.allow_tf32.
    filter_eps: skip vocabulary blocks whose probabilities all fall below it (and hold no target) in the gradient; None keeps it exact.
    strategy: "precompute" (default; three products: full vocabulary rows for one block of tokens, gradients made in the forward) or
    "recompute" (four products: online log-sum-exp in the forward, logits made again in the backward; the least memory between the two).
    """
    d = hidden.shape[-1]
    if shift:
        hidden, labels = hidden[..., :-1, :], labels[..., 1:]
    h, y = hidden.reshape(-1, d), labels.reshape(-1)
    keep = y != ignore_index
    h, y = h[keep], y[keep]
    if h.shape[0] == 0:
        return hidden.sum() * 0.0 + weight.sum() * 0.0
    cdt = compute_dtype or torch.promote_types(hidden.dtype, weight.dtype)
    n, v = h.shape[0], weight.shape[0]
    grads = torch.is_grad_enabled() and (h.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad))
    if strategy == "precompute" and grads:
        ct = min(n, chunk_tokens or max(16, int(block_mb * 2**20 // (4 * v)) // 16 * 16))   # full vocabulary rows: block_mb of them in fp32
        cv = min(v, chunk_vocab or 8192)
        return _FusedLinearCrossEntropyPrecompute.apply(h.contiguous(), weight, bias, y, reduction, cdt, fp32_precision, ct, cv, filter_eps)
    ct, cv = _sizes(n, v, d, block_mb, chunk_tokens, chunk_vocab)
    return _FusedLinearCrossEntropy.apply(h.contiguous(), weight, bias, y, reduction, cdt, fp32_precision, ct, cv, filter_eps)
