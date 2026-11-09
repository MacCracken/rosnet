# rosnet — Roadmap

> The sequencing: what ships next, in what order, against what gates. Current
> state lives in [`state.md`](state.md); what already shipped lives in
> [`CHANGELOG.md`](../../CHANGELOG.md).

**Shipped so far:** 0.1 scaffold → 0.2 GPU profile (ADR 0001) → **1.0.0 freeze**
(2026-07-04) → 1.1.0 conv2d / conv1d → 1.1.1 – 1.2.1 toolchain realigns →
**1.2.2** P(-1) hardening / optimization sweep
([audit](../audit/2026-11-09-audit.md)) → **1.3.0** the sweep's open ends (GPU
fixes validated on a device, untouched outputs on fallback, a soft shader-cache
cap, sticky dispatch failure, an aarch64 CI leg) → **1.4.0** GPU attention at
real head sizes (scores and dp tiled over hd: GPT-2's hd 64 runs on the device).

**Carried over (lands in attn11, not here):**
- attn11 moves to cyrius ≥ 6.6.19 before it takes any GPU bundle from 1.1.3 on
  (the distlib include prelude).
- With rosnet ≥ 1.4.0, attn11's `tests/gpu_attn_bwd.cyr` must turn its two
  `expect_fb(16, 64, 2)` / `expect_fb(16, 64, 1)` lines into `run_shape` checks:
  hd 32 and 64 now run on the device. With that change its 12-test suite passes
  against rosnet 1.4.0 (checked in a scratch copy).

## Rules for every 1.x release

- **The public surface is frozen** ([`docs/api.md`](../api.md)). A 1.x minor
  only adds; nothing listed there changes signature or meaning.
- **Optimizations keep the bits** ([ADR 0002](../adr/0002-kernel-optimizations-preserve-bits.md)):
  every output keeps its sequence of roundings, proven by the raw-bit golden
  dump and the oracles in `tests/reference.cyr`. Such work can ride any patch.
- **A change that moves bits is a numerical release**: its own minor, a
  CHANGELOG entry saying which outputs move and by how much, and a heads-up to
  the consumers whose suites compare bit patterns (attn11's bit-exact GPU
  class, byte-reproducible training).
- **A new stdlib leaf in a bundle's sidecar is consumer-facing**: say so in the
  CHANGELOG and check the six consumers' manifests first.

## 1.4.1 — AVX2 for the packed AXPY (decided, scheduled)

Bit-preserving (ADR 0002): a performance patch, not a numerical release. The
decision to adopt it was made after the 1.4.0 measurement.

- **What:** `linear_fwd` and `linear_bwd`'s dW (the same per-row AXPY) gain an
  AVX2 path behind `simd_has_avx2()`. It builds a 128-lane broadcast of
  x[m,k] with one `f64v256_scale` of a row of ones, then runs one
  `f64v256_fmadd` per 128-lane chunk. That lowers to `vmulpd` + `vaddpd`, not a
  fused op, so it is bit-identical to today's SSE2 `mulpd` + `addpd`. The
  `n4` split and the scalar tail stay where they are. aarch64 never takes the
  path, because the CPUID probe compiles out there.
- **Measured in 1.4.0:** bit-identical (0 differing elements at N = 768 and
  771) and ~9.5% faster at 64×768×768 (11.0 vs 12.1 ms, Ryzen 7 5800H,
  interleaved). A two-pass `scale` + `add` variant was slower and is not the
  plan.
- **Consumer-facing:** the `simd` stdlib leaf joins the CPU sidecar, and none
  of the six consumers declares it today. The CHANGELOG says so, and each
  consumer's build is checked before the cut.
- **Tests:** CI's runners have AVX2, so a test-only switch forces the SSE2
  path, and the suite and fuzz harness check *both* paths against the oracles
  in `tests/reference.cyr`. The golden dump must stay byte-identical on both.
- **Also measure, adopt only if it pays:** `t_scale` / `t_add_into` through
  `f64v256_scale` / `f64v256_add`.
## 1.x.x — demand-gated, unscheduled

Each waits for a consumer that needs it. None is breaking.

- **IEEE-754 wideners** (f32 / f16 / bf16 → f64): promote from anukūlana's
  importer once a second consumer (a GGUF parser or another importer) needs it.
- **Quantized dtypes:** these step outside the caller-guarantees contract (they
  read data, not caller pointers), so the loader must bounds-check like
  tula's / anukūlana's parsers (the 1.0 audit's standing guidance).
- **Numerical releases** (each its own minor under the rules above), when a
  consumer is throughput-blocked:
  - blocked / tiled matmul (a different k-order per output);
  - threads (per-thread partial sums);
  - a wider `linear_bwd` dx reduction (more lanes);
  - packed `t_axpy` (adds aarch64's fused-multiply-add divergence to it);
  - im2col-based conv.
- **Strided conv (SW > 1) — needs a different layout.** Measured in 1.4.0: the
  stride-1 row kernel with a per-tap gather was bit-identical but only ~10%
  faster at stride 2 and ~7% slower at stride 3 (short rows), because a gather
  costs about as much per element as the scalar multiply-add it replaces. Not
  adopted. A phase-split layout (each input row split once into SW strided
  runs) or im2col is what would pay.

## 2.0.0 backlog — breaking, only if a consumer forces it

Nothing here is planned. Each item would break the frozen 1.x contract, so it
waits until a real consumer need outweighs migrating six consumers.

- **Hot-path bounds / shape checking.** The audited substrate contract is
  caller-guarantees; checking every call is a contract change, and per-element
  checks would tax the loops rosnet exists to provide.
- **A shape-carrying tensor handle** instead of a raw pointer plus caller-held
  dims. It is what would make the checks above cheap (one check per call) and
  what broadcasting or views would need.
- **Error returns instead of exiting.** `t_alloc` exits on failure by contract;
  a library that returns errors changes every call site.
- **GPU control as API:** an explicit context or flags instead of the
  consumer-set globals `g_gpu` / `g_gpu_tc`, and fallback reason codes instead
  of a bare non-0.
- **One cross-target numerical contract.** Today x86_64 rounds the packed
  multiply-add twice and aarch64 once, so `linear_fwd` / `linear_bwd` differ in
  the last bits between them. Making them agree either costs aarch64 its fused
  path or moves every x86_64 result — a contract decision, not a patch.

## Out of scope

- A murti-style resident-model / runtime layer (hoosh's seam, per
  `agnosticos docs/development/planning/murti.md`).
- GPU provider work: devices, drivers, other vendors (mabda's surface, not
  rosnet's).
