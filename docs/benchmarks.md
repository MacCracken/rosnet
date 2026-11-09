# rosnet — Benchmarks

> 1.2.2 capture (2026-11-09), from the P(-1) sweep
> ([`docs/audit/2026-11-09-audit.md`](audit/2026-11-09-audit.md)). Harness:
> `tests/rosnet.bcyr` (stdlib `bench`, batch-timed), run exactly as CI does:
>
> ```sh
> ./scripts/bench-history.sh            # median of 3 runs, appended to bench-history.csv
> CYRIUS_DCE=1 cyrius bench tests/rosnet.bcyr   # one run
> ```
>
> Box: AMD Ryzen 7 5800H (Zen 3), x86_64 Linux, cyrius 6.7.18. Single-threaded,
> SSE2 packed f64 (2 lanes). Re-capture on a toolchain or hardware change, and
> compare rows only within one session on one host — the same binary moves a few
> percent between runs.

## The numbers — 1.2.1 → 1.2.2, same session

Both columns are the median of three runs of the same harness in one session
(`bench-history.csv`, stamps `2026-11-09T19:44:08Z` and `2026-11-09T20:12:46Z`).
Every 1.2.2 kernel returns the same bits as 1.2.1 (see *Reading them honestly*).

| Op | Shape | 1.2.1 | 1.2.2 | Speed-up | Rate (1.2.2) |
|----|-------|------:|------:|---------:|------|
| `t_axpy` | n=768 | 2.29 µs | 2.27 µs | — | scalar on purpose (below) |
| `t_scale` | n=768 | 1.71 µs | **0.19 µs** | 8.9× | |
| `t_add_into` | n=768 | 1.73 µs | **0.25 µs** | 6.8× | |
| `t_sum` | n=768 | 1.81 µs | 1.86 µs | — | order-defined, scalar |
| `t_copy` | n=768 | 1.12 µs | 1.14 µs | — | |
| `t_zero` | n=768×768 | 0.76 ms | 0.77 ms | — | |
| `linear_fwd` | 64×768×768 (GPT-2 projection, T=64) | 33.4 ms | **12.4 ms** | 2.7× | 37.7 M MAC → 3.0 G MAC/s |
| `linear_fwd` | 1×768×12564 (¼ LM-head row¹) | 9.07 ms | **4.08 ms** | 2.2× | full 50257-vocab row ≈ 16 ms |
| `linear_bwd` | 64×768×768 | 70.4 ms | **43.3 ms** | 1.6× | dx + dW = 75.5 M MAC → 1.7 G MAC/s |
| `conv2d_fwd` | 8×8×28×28 → 8, 3×3, pad 1 | 33.4 ms | **4.15 ms** | 8.0× | 3.61 M MAC → 0.87 G MAC/s |
| `conv2d_bwd` | same shape | 45.2 ms | **15.2 ms** | 3.0× | |
| `conv1d_fwd` | 4×16×1024 → 16, K=5, pad 2 | 48.5 ms | **3.64 ms** | 13.3× | 5.24 M MAC → 1.4 G MAC/s |
| `conv1d_bwd` | same shape | 66.4 ms | **18.7 ms** | 3.6× | |

¹ the full weight-tied head W (768×50257 f64 = 309 MB) exceeds the 256 MB alloc
cap, so the bench runs a ¼-vocab slice; the op is linear in N.

**Correction.** The v1.0/1.1 capture gave the conv2d shape as "36.1 M MAC →
~1.1 G MAC/s". The shape is 8·8·28·28·8·3·3 = **3.61 M** MAC, so 1.1's 32.4 ms
was ~0.11 G MAC/s — ten times slower than stated, and ten times slower than
`linear_fwd`, not "holding the linear-kernel rate". The conv1d row added in
1.2.2 showed the same ~0.1 G MAC/s and exposed it.

The v1.0 capture (cycc 6.4.2, 2026-07-04) for reference: `t_axpy` 2 µs,
`linear_fwd` 30.9 ms / 8.5 ms, `linear_bwd` 73.5 ms, `conv2d_fwd` 32.4 ms,
`conv2d_bwd` 43.3 ms. Those are a different toolchain and session; compare the
table above, not across captures.

## What changed in 1.2.2

- **`linear_fwd` and `linear_bwd`'s dW**: each `(row, k)` step is one
  `f64v_axpy(y_row, W_row, x[m,k], n4)` — the packed `y += α·x` builtin over the
  whole row — instead of a 4-wide `f64v_fmadd` per 4 elements after filling a
  4-wide broadcast buffer. Same split point (`n4`), same per-element operations.
- **`t_scale` / `t_add_into`**: `f64v_scale` / `f64v_add` over the even prefix,
  scalar for an odd last element.
- **conv**: the tap window is computed once per output position instead of
  bounds-testing every multiply-add, and base pointers are hoisted. For stride
  1 along W (every shape above), the forward pass runs along the output row:
  per tap, `tmp = x·w` (`f64v_scale`) then `acc += tmp` (`f64v_add`), in chunks
  of 128 outputs. The backward pass packs dW's products (`f64v_mul`, adds kept
  sequential) and runs dx along the row, walking kw downward so every dx element
  still receives its terms in ascending ow.
- Unchanged on purpose: `linear_bwd`'s dx (its 4-lane partial-sum shape is its
  bits; a wider call would mean more lanes), `t_sum` (sequential sum),
  `t_axpy` (see below), `t_copy` / `t_zero` (the stdlib `memcpy` / `memset` are
  byte loops — slower than the 8-byte loops here).

## Reading them honestly

- **Same bits.** Every change keeps each output's sequence of roundings. A
  15,302-line golden dump of raw f64 bit patterns (every CPU op at fixed seeds,
  incl. odd tails, rows of 128+ outputs, padding windows that miss the input,
  K = 0, no-bias paths) is byte-identical between 1.2.1 and 1.2.2, and the
  shipped 1.2.2 bundle gives the same dump under cyrius 6.6.2 (the consumers'
  pin). The suite checks each kernel against the 1.2.1 loops
  (`tests/reference.cyr`), and the fuzz harness does it for 15,000 random
  shapes per CI run.
- **x86_64 vs aarch64.** On x86_64 `f64v_axpy` / `f64v_fmadd` lower to `mulpd`
  then `addpd` (two roundings, exactly the scalar code). On aarch64 cyrius
  lowers both to a fused `FMLA` (one rounding), so `linear_fwd` and
  `linear_bwd`'s dx / dW differ from x86_64 in the last bits there — in 1.2.1
  as in 1.2.2. `t_axpy` stays scalar so it does not gain that divergence;
  `f64v_scale` / `f64v_add` / `f64v_mul` round once on both targets, so the
  BLAS-1 and conv changes are bit-identical everywhere.
- These are still **correctness-first kernels**, not tuned BLAS: SSE2 (2-lane)
  packed ops, no blocking/tiling, no AVX2, no threads. An OpenBLAS-class
  library is still an order of magnitude or more faster on this box. AVX2
  (`f64v256_*`, 4 lanes) would need a runtime CPU check and a new stdlib leaf
  in the bundle — left for a consumer that is throughput-blocked.
- The **GPU profile** (`dist/rosnet-gpu.cyr`, mabda) is benchmarked in its
  consumer (attn11's GPU suite, bit-exact/~1e-13 parity vs this CPU path as the
  oracle); it has no standalone numbers here because the profile's contract is
  parity + dispatch, and device throughput is a mabda/provider property.
