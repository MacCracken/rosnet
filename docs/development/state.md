# rosnet — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**1.2.2 — 2026-11-09 (P(-1) hardening / optimization / repair sweep).**
[`docs/audit/2026-11-09-audit.md`](../audit/2026-11-09-audit.md). Fixed a
stack-buffer overflow in `gpu_attn_core` (no hd cap; hd = 64 overran
`spv[8192]`), a wrong-result attention cache key, a leaking full shader cache,
`t_alloc(0)` exiting the process, and eleven LOW guard / CI / test gaps.
Kernels 1.6–13× faster with every output bit-identical to 1.2.1 (ADR 0002).
Tests 23 → 114 assertions + a real property fuzzer. No public signature
changed.

(**1.2.1** 2026-11-09 — cyrius → 6.7.18, tyche → 1.2.2. **1.2.0** 2026-11-09 — onto the cyrius 6.7 series: cyrius → 6.7.9, tyche →
1.2.0, tests run as `cyrius test tests`. **1.1.3** 2026-11-09 — cyrius → 6.6.20, tyche → 1.1.1; smaller sidecars; the
GPU bundle's reliance on the consumer's `puts` documented. **1.1.2** 2026-09-11 — cyrius → 6.6.2; fixed the GPU bundle's retired
`payload` accessor. **1.1.1** 2026-08-17 — cyrius → 6.5.27, tyche → 1.0.1.)

**1.1.0 — RELEASED 2026-07-05 (conv2d/conv1d).** The 1.1 additive: NCHW,
per-axis stride/pad, hand-derived FD-gated gradients (incl. strided/padded/
non-square configs), conv1d = exact H=1 wrappers; suite 7→**23**; dist
245→390 lines; benched ~0.11 G MAC/s (recorded at the time as ~1.1 — a 10×
MAC-count slip corrected in 1.2.2). **The modality axis is now
substrate-complete** (agnosticos `planning/multimodal-substrate.md`).

**1.0.0 — RELEASED 2026-07-04 (the substrate freeze).** No behavior change from
0.2.0: froze what six shipping consumers exercised unchanged since 0.1.1.
Public surface documented in [`docs/api.md`](../api.md) (CPU bundle + GPU
profile both frozen — mabda 4.x churn is provider-level, 4.0 added basic
NVIDIA, not the device/dispatch API). Benchmarks captured
([`docs/benchmarks.md`](../benchmarks.md), Ryzen 7 5800H), audit PASS
([`docs/audit/2026-07-04-audit.md`](../audit/2026-07-04-audit.md), 0 code
changes — the no-bounds-checks caller-guarantees contract audited as
deliberate). All six v1.0 criteria met.

(**0.2.0** 2026-06-21 — GPU backend as the mabda-gated `[lib.gpu]` profile,
ADR 0001, extracted whole from attn11 M18/M19, validated bit-exact/~1e-13 vs
the CPU oracle in attn11's 12-test GPU suite. **0.1.1** — scaffolded
2026-06-11 via `cyrius init`.)

## Toolchain

- **Cyrius pin**: `6.7.18` (in `cyrius.cyml [package].cyrius`), the latest
  release, since 2026-11-09. `lib/` byte-matches the 6.7.18 snapshot (113
  files).
- **Consumer floor**: the CPU bundle compiles (and matches the golden dump) at
  cyrius 6.6.2; the GPU bundle needs **≥ 6.6.19** in the consumer (its
  distlib-written `include "lib/mabda.cyr"` prelude fails on 6.6.2 – 6.6.18).
- **Bump procedure:** edit the pin → `cyrius lib sync --full` (cyrius refuses a
  moved pin until this runs) → `cyrius deps` → if a git dep's tag moved, delete
  `cyrius.lock` and run `cyrius deps` again (the lock keeps the old pin
  fail-closed) → `cyrius distlib` + `cyrius distlib gpu`. Commit the pin,
  `lib/` and `cyrius.lock` together, or CI's `cyrius deps` fails.
- `lib/` is exactly the 6.7.18 snapshot plus `lib/tyche.cyr`; `cyrius.lock`
  records 114 files. (`lib/agnosys.cyr`, a scaffold-era orphan `lib sync` never
  pruned, was removed in 1.2.2 with the maintainer's go-ahead.) After a pin bump,
  check for files the new snapshot no longer ships: `lib sync --full` copies
  but does not prune.
- Earlier: 6.2.11 → 6.5.27 (2026-08-17, ended the deliberate deep-lag),
  6.5.27 → 6.6.2 (2026-09-11), 6.6.2 → 6.6.20 and 6.6.20 → 6.7.9
  (2026-11-09).

## Source

- `src/tensor.cyr` — storage + BLAS-1 (`t_alloc`/`t_zero`/`t_fill`/`t_copy`,
  `tget`/`tset`/`ff`, `t_axpy`/`t_add_into`/`t_scale`/`t_sum`, `f64_is_finite`,
  `t_randn`). `t_scale` / `t_add_into` packed (`f64v_scale` / `f64v_add`).
- `src/linear.cyr` — dense matmul + gradient (`linear_fwd`/`linear_bwd`);
  `linear_fwd` and dW one `f64v_axpy` per row step; dx a 4-lane `f64v_fmadd` chain.
- `src/conv.cyr` — conv2d/conv1d fwd+bwd (1.1 additive; per-axis stride/pad);
  stride-1 (along W) fwd / bwd run along the output row with packed ops.
- `src/gpu.cyr` — GPU backend (`[lib.gpu]` profile, ADR 0001; mabda-gated,
  Linux-only), extracted whole from attn11 (M18/M19). Shader cache 1024
  entries; every key field range-checked.
- `src/main.cyr` — smoke demo (`main()` + `SYS_EXIT`; excluded from the bundle).

CPU `[lib]` bundle → `dist/rosnet.cyr` (640 lines, v1.2.2; sidecar
`dist/rosnet.deps` = syscalls, string, fmt, alloc); GPU profile →
`dist/rosnet-gpu.cyr` (2406 lines, v1.2.2; sidecar = mabda, math, result,
ganita, syscalls, string). Regenerate with `cyrius distlib --all`; verify with
`cyrius distlib --all --check`. Since 1.2.2 the GPU bundle needs no symbol from
its consumer beyond mabda.

## Tests / benches

- `tests/rosnet.tcyr` — CPU suite: exact BLAS-1 checks, central
  finite-difference grad-checks of `linear_bwd` / `conv2d_bwd`, conv
  known-answer, conv1d parity (fwd + bwd), storage, and every optimized kernel
  against the plain-loop oracles in `tests/reference.cyr`. **54/54**.
- `tests/gpu_host.tcyr` — the GPU profile's host side, no device: cache
  capacity / full-cache behaviour, key fields, every SPIR-V builder within
  `spv[8192]` at its guard limits. **60/60**.
- `tests/rosnet.fcyr` — seeded property fuzzer: 15,000 random shapes per run
  (±0 / subnormal / huge / ±Inf inputs) against the oracles; ~4 s.
- `tests/rosnet.bcyr` — benchmarks (every BLAS-1 op, linear, conv2d, conv1d);
  `scripts/bench-history.sh` appends medians to `bench-history.csv`; numbers in
  [`docs/benchmarks.md`](../benchmarks.md).

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math
- **tyche** (`1.2.2`, commit `cdb74e2`; the latest release, built against
  cyrius 6.7.18) — deterministic statistical PRNG; supplies
  `rng_seed`/`rng_normal` for `t_randn`. The only non-stdlib dep; pure
  tensor/matmul ops need none. (The GPU profile additionally needs **mabda**,
  supplied by the consumer — rosnet declares no top-level mabda dep.)

## Consumers (six, all green at the freeze)

**attn11** (GPU backend consumer + oracle suite) · **tarka** · **tentib** ·
**prajna** · **rupantara** (the transformer-forward lib) · **anukūlana**
(Type-3 import — the real-GPT-2 forward runs on this substrate).

## Next

Post-1.0 additive lane (demand-gated): the f32/f16/bf16 wideners promotion;
CPU optimizations that change bits (blocking/tiling, threads, AVX2, a wider dx
reduction — ADR 0002 makes each a numerical change) when a consumer is
throughput-blocked; GPU attention for head dims above ~26 (tile the hd loop).
See [`roadmap.md`](roadmap.md).
