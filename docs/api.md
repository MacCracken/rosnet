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
| `t_alloc(n)` | heap-allocate n f64 (8n bytes); **fail-loud** (prints + exits) on out-of-memory; ≤ 256 MB per alloc (allocator cap) |
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
| `linear_fwd(x, W, b, y, M, K, N)` | `y(M,N) = x(M,K) · W(K,N) [+ b(N)]`; `b == 0` means no bias; 4-wide-SIMD dot, tree reduction |
| `linear_bwd(x, W, dy, dx, dW, db, M, K, N)` | writes `dx = dy·Wᵀ`; **accumulates** `dW += xᵀ·dy`, `db += Σdy` (caller zeroes before a fresh pass); `db == 0` means no bias grad. FD-gated (central differences, rel err < 1e-5) |

### Convolution (added + frozen at 1.1.0, 2026-07-05)

| Symbol | Contract |
|--------|----------|
| `conv2d_out_dim(in, K, S, P)` | `(in + 2P − K)/S + 1` |
| `conv2d_fwd(x, Wf, b, y, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW)` | NCHW ⊛ (Cout,Cin,KH,KW) [+ b(Cout)]; per-axis stride/zero-padding; `b == 0` no bias |
| `conv2d_bwd(x, Wf, dy, dx, dW, db, …same dims…)` | dx **written** (0 = skip); dW/db **accumulate**; FD-gated incl. stride/pad/non-square |
| `conv1d_fwd/bwd(x, Wf, [b/dy…], N, Cin, L, Cout, K, S, P)` | exact H=1 wrappers, bit-identical to the 2D path |

### Guards / init

| Symbol | Contract |
|--------|----------|
| `f64_is_finite(x)` | 1 iff x is neither NaN nor ±Inf (bit-pattern check) |
| `t_randn(p, n, std)` | N(0, std²) fill via **tyche** `rng_normal` — the ONE symbol needing `[deps.tyche]`; seed with `rng_seed(s)` for reproducibility |

## GPU profile — `dist/rosnet-gpu.cyr` (`[lib.gpu]`, ADR 0001)

Consumer imports the GPU bundle **and** declares `[deps.mabda]` (Linux-only).
mabda's device/dispatch symbols ship unresolved by design. Every op
**self-falls-back to CPU** (returns non-0) when the device is unavailable —
the gates and counters make engagement observable, never assumed.

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

Bit-exact class: `gpu_adam_step(params, grads, m, v, NP, lr, step)` ·
`gpu_head_bwd(D_logits, emb, A_f, D_f, gemb, T, V, C)` ·
`gpu_ln_bwd(x, gamma, dy, mean, rstd, dx, dgamma, dbeta, M, C)` ·
`gpu_rope_apply(X, cs, rows, width, num_heads, hdp, dir)`

Tolerance class (`g_gpu_tc`): `gpu_matmul_fwd(x, W, b, y, M, K, N)` ·
`gpu_ln_fwd(x, gamma, beta, y, mean, rstd, M, C)` · `gpu_gelu_fwd(x, y, n)` ·
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
