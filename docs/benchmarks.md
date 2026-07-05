# rosnet — Benchmarks

> v1.0 capture (2026-07-04). Harness: `tests/rosnet.bcyr` (stdlib `bench`,
> batch-timed). Reproduce:
>
> ```sh
> cyrius build tests/rosnet.bcyr build/bench && ./build/bench
> ```
>
> Box: AMD Ryzen 7 5800H (Zen 3), x86_64 Linux, cycc 6.4.2. Single-threaded,
> scalar+4-wide-SIMD f64 (`f64v_fmadd` inner loops). Numbers are avg over the
> stated iters; re-capture on toolchain or hardware change.

## The v1.0 numbers

| Op | Shape | Avg | Derived |
|----|-------|-----|---------|
| `t_axpy` | n=768 | **2 µs** | ~0.4 GFLOP/s BLAS-1 (bandwidth-shaped) |
| `linear_fwd` | 64×768×768 (GPT-2 projection, T=64) | **30.9 ms** | 37.7 M MAC → ~1.2 G MAC/s (~2.4 GFLOP/s) |
| `linear_fwd` | 1×768×12564 (¼ LM-head row¹) | **8.5 ms** | full 50257-vocab head row ≈ **34 ms** |
| `linear_bwd` | 64×768×768 | **73.5 ms** | ≈ 2.4× fwd (dx + dW + db — three passes) |
| `conv2d_fwd` (1.1) | 8×8×28×28→8, 3×3, pad 1 | **32.4 ms** | 36.1 M MAC → ~1.1 G MAC/s (direct loops hold the linear-kernel rate) |
| `conv2d_bwd` (1.1) | same shape | **43.3 ms** | ≈ 1.3× fwd (dx scatter + dW gather + db) |

¹ the full weight-tied head W (768×50257 f64 = 309 MB) exceeds the 256 MB alloc
cap, so the bench runs a ¼-vocab slice; the op is linear in N.

## Reading them honestly

- These are **correctness-first reference kernels**, not tuned BLAS: row-major
  loops + a 4-lane `f64v_fmadd` dot with tree reduction, no blocking/tiling, no
  threads. An optimized BLAS (OpenBLAS-class) is ~1–2 orders faster on the same
  box — that gap is the known cost of a from-scratch sovereign substrate, and
  closing it is post-1.0 optimization work (or the GPU profile's job).
- The consumer-observed anchor: anukūlana's full GPT-2-small forward (12 layers,
  T=8, incl. the 50257-vocab head each position) runs ~3.7 s on this box —
  consistent with these per-op numbers.
- The **GPU profile** (`dist/rosnet-gpu.cyr`, mabda) is benchmarked in its
  consumer (attn11's GPU suite, bit-exact/~1e-13 parity vs this CPU path as the
  oracle); it has no standalone numbers here because the profile's contract is
  parity + dispatch, and device throughput is a mabda/provider property.
