# rosnet — Public API (1.x FROZEN)

> **Frozen at v1.0.0 (2026-07-04).** Every symbol below is stable for the 1.x
> line: signatures, semantics, and the memory contract do not change; additions
> land as minor versions (additive only). Symbols prefixed `_` — and everything
> not listed here (the GPU pool, shader cache, SPIR-V emitters, dispatch
> helpers) — are **internal and out-of-freeze**.
>
> Frozen after six shipping consumers exercised the surface unchanged since
> 0.1.1: attn11, tarka, tentib, prajna, rupantara, anukūlana.

## The memory contract (applies to every op)

A tensor is a heap pointer to a flat, **row-major** run of f64 values (IEEE-754
bit patterns carried in `i64`). **Shapes live in the caller.** rosnet performs
**no bounds or shape checks** — this is the substrate contract, deliberate for a
hot-path BLAS layer: the caller guarantees every pointer covers its stated
extent and dims are positive. (See `docs/audit/2026-07-04-audit.md` for the
audited rationale.)

## CPU bundle — `dist/rosnet.cyr` (`[lib]`)

### Storage / element access

| Symbol | Contract |
|--------|----------|
| `t_alloc(n)` | heap-allocate n f64 (8n bytes), zeroed; **never returns 0**. Out-of-memory, `n < 0`, or `n` whose byte size would overflow → prints to **stderr** and exits 1; ≤ 256 MB per alloc (allocator cap). `n == 0` returns a valid, distinct pointer (an empty tensor) — before 1.2.2 it exited as "out of memory" |
| `t_zero(p, n)` · `t_fill(p, n, v)` · `t_copy(dst, src, n)` | element-wise over n f64 |
| `tget(p, i)` → f64 · `tset(p, i, v)` | single-element load/store at index i |
| `ff(n)` | f64 from an integer literal (`f64_from` shorthand) |

### BLAS-1

| Symbol | Contract |
|--------|----------|
| `t_axpy(dst, src, a, n)` | `dst += a·src` |
| `t_add_into(dst, src, n)` | `dst += src` |
| `t_scale(dst, n, s)` | `dst *= s` |
| `t_sum(p, n)` → f64 | Σ p[i] |

### BLAS-3 — matmul + its gradient (the reason rosnet exists)

| Symbol | Contract |
|--------|----------|
| `linear_fwd(x, W, b, y, M, K, N)` | `y(M,N) = x(M,K) · W(K,N) [+ b(N)]`; `b == 0` means no bias. Each `y[m,n]` is `b[n]` plus `x[m,k]·W[k,n]` added in k order (a packed AXPY over n) |
| `linear_bwd(x, W, dy, dx, dW, db, M, K, N)` | writes `dx = dy·Wᵀ` (a 4-lane partial-sum dot: lane j sums n ≡ j mod 4, then `(l0+l1)+(l2+l3)`, then the tail); **accumulates** `dW += xᵀ·dy` (k-order AXPY over m), `db += Σdy` (caller zeroes before a fresh pass); `db == 0` means no bias grad. FD-gated (central differences, rel err < 1e-5) |

### Convolution (added + frozen at 1.1.0, 2026-07-05)

| Symbol | Contract |
|--------|----------|
| `conv2d_out_dim(in, K, S, P)` | `(in + 2P − K)/S + 1` |
| `conv2d_fwd(x, Wf, b, y, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW)` | NCHW ⊛ (Cout,Cin,KH,KW) [+ b(Cout)]; per-axis stride/zero-padding; `b == 0` no bias |
| `conv2d_bwd(x, Wf, dy, dx, dW, db, …same dims…)` | dx **written** (0 = skip); dW/db **accumulate**; FD-gated incl. stride/pad/non-square |
| `conv1d_fwd/bwd(x, Wf, [b/dy…], N, Cin, L, Cout, K, S, P)` | exact H=1 wrappers, bit-identical to the 2D path |

### Numerical reproducibility

Every op's summation order is fixed and documented above, so results are
deterministic run to run. Across targets: on **x86_64** the packed bodies round
exactly like the scalar code (`mulpd` then `addpd`), so the vectorized kernels
equal the plain loops bit for bit. On **aarch64** cyrius lowers the packed
multiply-add (`linear_fwd`, `linear_bwd`'s dx / dW) to a fused `FMLA`, one
rounding, so those differ from x86_64 in the last bits. `t_scale`,
`t_add_into`, the conv ops and every scalar op round identically on both. A NaN
produced from two NaN operands may carry either operand's payload.

### Guards / init

| Symbol | Contract |
|--------|----------|
| `f64_is_finite(x)` | 1 iff x is neither NaN nor ±Inf (bit-pattern check) |
| `t_randn(p, n, std)` | N(0, std²) fill via **tyche** `rng_normal` — the ONE symbol needing `[deps.tyche]`; seed with `rng_seed(s)` for reproducibility |

## GPU profile — `dist/rosnet-gpu.cyr` (`[lib.gpu]`, ADR 0001)

Consumer imports the GPU bundle **and** declares `[deps.mabda]` (Linux-only).
mabda's device/dispatch symbols ship unresolved by design (since 1.2.2 the
bundle needs nothing else from the consumer — before, it called a `puts` the
consumer had to define). Every op **self-falls-back to CPU** (returns non-0)
when the device is unavailable or the shape is outside what it handles: a
non-positive dim; a dim off its tile multiple; a buffer past `GPU_BO_MAX()`; a
grid past mabda's dispatch cap; a dim too large for its shader-cache key field;
attention with `C % nh != 0` or a head dim over 32; or a full shader cache
(1024 kernels). The gates and counters make engagement observable, never
assumed.

**On a non-0 return, the outputs may be partially written** (e.g. `gpu_ln_fwd`'s
mean/rstd, `gpu_linear_bwd`'s dx, `gpu_head_bwd`'s D_f, `gpu_attn_core_bwd`'s
dQ/dV can be filled before a later stage falls back). The caller's CPU op must
recompute **every** output — as attn11's fallbacks do. The accumulated outputs
(`gpu_linear_bwd`'s dW / db, `gpu_head_bwd`'s gemb, `gpu_ln_bwd`'s dgamma /
dbeta) are written only after the last stage that can fall back, so a fallback
never adds a gradient twice.

**Provider note (orthogonal to this freeze):** the ops are validated bit-exact /
~1e-13 vs the CPU oracle on **AMD GFX9** (attn11's 12-test GPU suite); mabda 4.0
added basic **NVIDIA** support — provider coverage evolves under mabda without
touching this API.

### Gates (consumer-set globals)

| Symbol | Contract |
|--------|----------|
| `g_gpu` | 1 = dispatch bit-exact-classified ops on device |
| `g_gpu_tc` | 1 = ALSO dispatch tolerance-classified ops (~1e-13, breaks byte-identity); requires `g_gpu` |

### Ops (return 0 = ran on device; non-0 = caller falls back to CPU)

Bit-exact class (`g_gpu`): `gpu_matmul_fwd(x, W, b, y, M, K, N)` ·
`gpu_ln_fwd(x, gamma, beta, y, mean, rstd, M, C)` ·
`gpu_adam_step(params, grads, m, v, NP, lr, step)` ·
`gpu_head_bwd(D_logits, emb, A_f, D_f, gemb, T, V, C)` ·
`gpu_ln_bwd(x, gamma, dy, mean, rstd, dx, dgamma, dbeta, M, C)` ·
`gpu_rope_apply(X, cs, rows, width, num_heads, hdp, dir)`

Tolerance class (`g_gpu_tc`): `gpu_gelu_fwd(x, y, n)` ·
`gpu_gelu_bwd(x, dy, dx, n)` · `gpu_head_fwd(f, emb, logits, T, V, C)` ·
`gpu_linear_bwd(x, W, dy, dx, dW, db, M, K, N)` ·
`gpu_attn_core(Qc, Kc, Vc, Pc, concat, T, C, nh, nkv)` ·
`gpu_attn_core_bwd(Qc, Kc, Vc, Pc, dconcat, dQ, dK, dV, dprow, T, C, nh, nkv)`

### Observability

`gpu_probe()` (device availability) and the engagement counters — proof the
device actually ran, one per op: `gpu_mm_count` · `gpu_ln_count` ·
`gpu_gelu_count` · `gpu_head_count` · `gpu_attn_count` · `gpu_adam_count` ·
`gpu_linbwd_count` · `gpu_headbwd_count` · `gpu_lnbwd_count` ·
`gpu_attnbwd_count` · `gpu_rope_count`.

## Out of freeze

- Everything `_`-prefixed (pool, cache, SPIR-V emitters, VA constants).
- `GPU_TK()` / `GPU_BO_MAX()` tuning constants (may change with mabda arcs).
- The smoke `main()` in `src/main.cyr` (not in either bundle).
- Future additive surface (quantized dtypes, pooling, im2col/SIMD conv
  optimization) lands as 1.x minors. (conv2d/conv1d landed at 1.1.0.)
