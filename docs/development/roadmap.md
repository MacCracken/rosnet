# rosnet — Roadmap

> The sequencing: what ships next, in what order, against what gates. Current
> state lives in [`state.md`](state.md); what already shipped lives in
> [`CHANGELOG.md`](../../CHANGELOG.md).

**Shipped so far:** 0.1 scaffold → 0.2 GPU profile (ADR 0001) → **1.0.0 freeze**
(2026-07-04) → 1.1.0 conv2d / conv1d → 1.1.1 – 1.2.1 toolchain realigns →
**1.2.2** P(-1) hardening / optimization sweep
([audit](../audit/2026-11-09-audit.md)) → **1.3.0** the sweep's open ends: GPU
fixes validated on a device, fallbacks leave outputs untouched, a soft shader-
cache cap, sticky dispatch failure, an aarch64 CI leg.

**Carried over (lands in attn11, not here):** attn11 moves to cyrius ≥ 6.6.19
before it takes any GPU bundle from 1.1.3 on (the distlib include prelude). Its
12-test GPU suite already passes against rosnet 1.3.0 at cyrius 6.7.18.

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

## 1.4.x — bit-preserving speed

Each item keeps every output's rounding sequence (ADR 0002), so it is a
performance minor, not a numerical one.

- **GPU attention for real head sizes.** The untiled scores (forward) and dp
  (backward) kernels exceed mabda's 256-id cap — measured on a GFX9 device at
  cyrius 6.7.18 they compile only up to hd 11 and hd 17 — so GPT-2's hd = 64,
  and even hd = 16, fall back today. Tile their hd loops as attn11 1.9.0 tiled
  the T loops, then lift the `hd ≤ 32` guards (keeping every builder within
  `spv[8192]`, which `tests/gpu_host.tcyr` pins).
- **AVX2 path** for the packed kernels: `f64v256_fmadd` / `_scale` / `_add`
  lower to `vmulpd` + `vaddpd` — the same two roundings as SSE2 — behind
  `simd_has_avx2()`. That adds the `simd` stdlib leaf to the CPU sidecar, so it
  needs the consumer check above.
- **Strided conv (SW > 1) fast path:** gather each input run into a contiguous
  row, then the same packed scale + add as the stride-1 path. Same order, same
  bits.

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
