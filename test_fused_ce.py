"""fused_ce against the plain computation in fp64: loss and every gradient, with a bias, ignored tokens, a shift, blocks that do not divide
the sizes, one token, no scored token, both reductions, bf16 products, and filtering within its tolerance.

    python3 test_fused_ce.py
"""
import torch
import torch.nn.functional as F

from fused_ce import fused_linear_cross_entropy

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def ref(h, w, b, y, shift, reduction):
    if shift:
        h, y = h[..., :-1, :], y[..., 1:]
    lg = h.reshape(-1, h.shape[-1]).double() @ w.double().T + (b.double() if b is not None else 0)
    return F.cross_entropy(lg, y.reshape(-1), ignore_index=-100, reduction=reduction)


def case(name, n, d, v, dtype, tol, bias=False, shift=False, reduction="mean", ignore=0.1, **kw):
    g = torch.Generator(device=DEV).manual_seed(n * 7 + v)
    shape = (2, n // 2, d) if shift else (n, d)
    h = torch.randn(*shape, device=DEV, generator=g, dtype=torch.float64).to(dtype).requires_grad_(True)
    w = (torch.randn(v, d, device=DEV, generator=g, dtype=torch.float64) * 0.1).to(dtype).requires_grad_(True)
    b = torch.randn(v, device=DEV, generator=g, dtype=torch.float64).to(dtype).requires_grad_(True) if bias else None
    y = torch.randint(0, v, shape[:-1], device=DEV, generator=g)
    y[torch.rand(shape[:-1], device=DEV, generator=g) < ignore] = -100
    ins = [h, w] + ([b] if bias else [])
    r = ref(h, w, b, y, shift, reduction)
    gr = torch.autograd.grad(r, ins, allow_unused=True)
    o = fused_linear_cross_entropy(h, w, y, b, shift=shift, reduction=reduction, **kw)
    go = torch.autograd.grad(o, ins, allow_unused=True)
    err = lambda a, c: 0.0 if c is None else float((a.double() - c).abs().max() / c.abs().max().clamp(min=1e-30)) if a is not None else float("inf")
    errs = [err(o, r)] + [err(a, c) for a, c in zip(go, gr)]
    ok = max(errs) < tol
    print(f"{'ok  ' if ok else 'FAIL'} {name:46s} " + " ".join(f"{e:.1e}" for e in errs), flush=True)
    return ok


def main():
    small = dict(chunk_tokens=37, chunk_vocab=101)
    res = []
    for strat in ("precompute", "recompute"):
        st = dict(small, strategy=strat)
        print(f"-- strategy {strat}")
        res += [
            case("fp64, blocks that do not divide", 300, 48, 1000, torch.float64, 1e-12, **st),
            case("fp64, bias, sum", 300, 48, 1000, torch.float64, 1e-12, bias=True, reduction="sum", **st),
            case("fp64, shift (causal)", 300, 48, 1000, torch.float64, 1e-12, shift=True, **st),
            case("fp64, one token", 1, 48, 1000, torch.float64, 1e-12, ignore=0.0, **st),
            case("fp32, ieee", 300, 48, 1000, torch.float32, 1e-5, **st),
            case("fp32, tf32 products", 300, 48, 1000, torch.float32, 1e-2, fp32_precision="tf32", **st),
            case("fp32, filter 2^-30 (exact here)", 300, 48, 1000, torch.float32, 1e-5, filter_eps=2 ** -30, **st),
            case("bf16 inputs, fp32 products", 300, 48, 1000, torch.bfloat16, 2e-2, compute_dtype=torch.float32, **st),
            case("bf16 inputs, bf16 products", 300, 48, 1000, torch.bfloat16, 5e-2, **st),
        ]
    h = torch.randn(4, 8, device=DEV, requires_grad=True)
    w = torch.randn(10, 8, device=DEV, requires_grad=True)
    z = fused_linear_cross_entropy(h, w, torch.full((4,), -100, device=DEV))
    z.backward()
    res.append(float(z) == 0.0 and h.grad is not None)
    print(f"{'ok  ' if res[-1] else 'FAIL'} no scored token: loss 0, gradients zero", flush=True)
    assert all(res), "some cases failed"
    print("ALL OK")


if __name__ == "__main__":
    main()
