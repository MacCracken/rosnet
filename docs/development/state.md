# rosnet — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**1.4.1 — 2026-11-09 (AVX2 for the packed kernels).** `linear_fwd`, linear_bwd's
dW, `t_scale` and `t_add_into` run 4 lanes wide on an AVX2 CPU (`vmulpd` +
`vaddpd`, the same bits; aarch64 unchanged). A mixed training step is 3.1%
faster on a Ryzen 7 5800H. The `simd` leaf joins the CPU sidecar (the bundle
includes it; it compiles under consumers' cyrius 6.6.2). No API change.

**1.4.0 — 2026-11-09 (GPU attention at real head sizes).** The forward scores
and backward dp kernels run in hd tiles of 8, so attention runs on the device at
any hd below 128 — GPT-2's 64 included (1.3.0 fell back above 11 / 17). Same
bits wherever 1.3.0 already ran (device dump); CPU unchanged. No API change.
attn11's backward test needs two `expect_fb` lines flipped (see the roadmap).

**1.3.0 — 2026-11-09 (the 1.2.2 sweep's open ends).** GPU fixes validated on
a GFX9 device (attn11's 12-test suite passes against this release); every GPU
fallback leaves the caller's outputs untouched; a full shader cache is a soft
limit (transient modules released after their dispatch); a failing dispatch is
sticky; an aarch64 CI leg under qemu. CPU results and generated shaders
byte-identical to 1.2.2. No API change.

(**1.2.2** 2026-11-09 — the P(-1) sweep ([audit](../audit/2026-11-09-audit.md)):
a GPU stack overflow, a wrong-result attention key, a leaking cache, `t_alloc(0)`
exiting; kernels 1.6–13× faster, bit-identical. **1.2.1** 2026-11-09 — cyrius → 6.7.18, tyche → 1.2.2. **1.2.0** 2026-11-09 — onto the cyrius 6.7 series: cyrius → 6.7.9, tyche →
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
  entries, a soft limit (kernels past it are compiled, dispatched once and
  released); every key field range-checked; all dispatches go through
  `_gpu_dispatch` (sticky on failure); outputs read back after the last dispatch.
- `src/main.cyr` — smoke demo (`main()` + `SYS_EXIT`; excluded from the bundle).

CPU `[lib]` bundle → `dist/rosnet.cyr` (726 lines, v1.4.1; sidecar
`dist/rosnet.deps` = syscalls, string, fmt, alloc, simd); GPU profile →
`dist/rosnet-gpu.cyr` (2551 lines, v1.4.1; sidecar = mabda, math, result,
ganita, syscalls, string). Regenerate with `cyrius distlib --all`; verify with
`cyrius distlib --all --check`. Since 1.2.2 the GPU bundle needs no symbol from
its consumer beyond mabda.

## Tests / benches

- `tests/rosnet.tcyr` — CPU suite: exact BLAS-1 checks, central
  finite-difference grad-checks of `linear_bwd` / `conv2d_bwd`, conv
  known-answer, conv1d parity (fwd + bwd), storage, and every optimized kernel
  against the plain-loop oracles in `tests/reference.cyr`, on both the AVX2 and
  the SSE2 path (`_rosnet_force_sse2`, test-only), plus the two paths compared
  bit for bit. **70/70** on x86_64, **79/79** on aarch64 (qemu; the linear cases
  take the tolerance branch).
- `tests/gpu_host.tcyr` — the GPU profile's host side, no device: cache
  capacity / full-cache behaviour, key fields, every SPIR-V builder within
  `spv[8192]` at its guard limits (the tiled attention builders at hd 127),
  transient lifetime, sticky dispatch. **80/80** (x86_64 and aarch64).
- `tests/gpu_device.tcyr` — the GPU ops on a real device (without one, e.g. on
  CI, 7 assertions of the no-device fallback contract instead):
  bit-exact matmul, the attention cache-key pair, untouched outputs on a
  late-stage fallback, the soft cache cap, forward / backward attention at hd
  12–64 and GPT-2 small. **60/60** on a Radeon Vega (GFX9).
- `tests/rosnet.fcyr` — seeded property fuzzer: 15,000 random shapes per run
  (±0 / subnormal / huge / ±Inf inputs) against the oracles; ~4 s.
- `tests/rosnet.bcyr` — benchmarks (every BLAS-1 op, linear, conv2d, conv1d);
  `scripts/bench-history.sh` appends medians to `bench-history.csv`; numbers in
  [`docs/benchmarks.md`](../benchmarks.md).

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math, simd (the AVX2 probe, 1.4.1)
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

Nothing is scheduled past 1.4.1. Next come the demand-gated 1.x minors
(wideners, quantized dtypes, numerical releases, strided conv via a different
layout) and a 2.0.0 backlog nothing yet forces. Carried over to attn11: cyrius
≥ 6.6.19, and two `expect_fb` lines → `run_shape` for rosnet ≥ 1.4.0. See
[`roadmap.md`](roadmap.md).
