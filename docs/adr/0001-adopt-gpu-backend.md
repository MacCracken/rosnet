# 0001 — Adopt the GPU backend as a mabda-gated `[lib.gpu]` profile

**Status**: Accepted
**Date**: 2026-06-21

## Context

rosnet is the AGNOS tensor/numeric core — CPU SIMD matmul + hand-derived gradient — consumed as the
`dist/rosnet.cyr` bundle (`[lib].modules = ["src/tensor.cyr", "src/linear.cyr"]`). It has **consumers
beyond attn11**, and its CPU bundle must stay **dependency-clean**: pure tensor ops need no GPU,
no Linux-only syscalls, no 1 MB device library.

attn11 built a GPU compute backend over M18/M19 (`src/gpu.cyr`) on **mabda**'s native-AMD f64
SPIR-V→GFX9 path — proven bit-exact / ~1e-13-tolerance against the CPU oracle, the full training step
on-device. Per [attn11 ADR 0017](https://github.com/MacCracken/attn11/blob/main/docs/adr/0017-extract-gpu-backend-to-rosnet.md)
it is being extracted into rosnet as the GPU sibling of rosnet's SIMD path (layered **on** mabda's
primitives, never in mabda — see attn11 ADR 0016). mabda is the GPU *foundation* (device / buffers /
compute-dispatch) and is **Linux-only** (DRM/PM4); forcing it onto rosnet's CPU consumers is
unacceptable. **The decision: how does rosnet host a GPU backend without contaminating its CPU core?**

## Decision

Add `src/gpu.cyr` — the whole backend from attn11 (the GPU foundation: device/ctx, the GTT buffer
pool, the hand-emitted SPIR-V primitives, the tiled-matmul builders, the shader cache, dispatch; the
generic tensor ops `matmul`/`ln`/`gelu`/`adam`/`linear_bwd`/`ln_bwd`; and, per attn11 ADR 0017's
whole-backend choice, the transformer ops `attn_core`/`head`/`rope`) — and expose it as a **separate
`[lib.gpu]` profile** that `cyrius distlib` bundles into **`dist/rosnet-gpu.cyr`**. The default
`[lib]` bundle (`dist/rosnet.cyr`, tensor + linear) is **unchanged and mabda-free**; only a consumer
that opts into `dist/rosnet-gpu.cyr` pulls mabda.

`dist/rosnet-gpu.cyr` ships with mabda symbols **unresolved** — the consumer-included-bundle pattern
`cyrius distlib` already supports (stdlib + mabda are supplied by the consumer). rosnet does **not**
declare mabda in its top-level `[deps]` (cyrius auto-prepends a dep's `modules` on *every* target,
which would contaminate the CPU bundle and the AGNOS build); rosnet's own GPU tests include mabda
locally, and are device-dependent (skip cleanly without an AMD f64 GPU). Ships as **rosnet 0.2.0** —
a minor: additive (a new optional bundle), the CPU core untouched and byte-identical.

## Consequences

- **Positive** — rosnet gains a reusable GPU tensor backend; the native-AMD f64 path (and future
  vendors — the Nvidia bring-up) lives once, in the numeric core, for every consumer; the CPU bundle
  stays clean (no mabda, no Linux-only surface, no size bloat for CPU-only users).
- **Negative** — rosnet now carries a Linux-only **optional** bundle; per attn11 ADR 0017 the GPU
  bundle temporarily holds **transformer-specific** kernels (attention / RoPE / head) that are not
  generic tensor ops — a boundary blur the numeric core would rather not own; a GPU kernel fix is now
  a rosnet release.
- **Neutral** — a follow-up may re-home the transformer ops to the consumer once the GPU foundation
  has a second consumer to harden its public API against; the GPU bundle's test lane mirrors attn11's
  (device-dependent, out of the default gate).

## Alternatives considered

- **Put the GPU backend in mabda** — rejected upstream (attn11 ADR 0016): it contaminates mabda's
  device-primitive purpose with tensor/NN math.
- **One bundle (GPU folded into the default `[lib]`)** — would force mabda + the Linux-only surface
  onto every CPU consumer and break the AGNOS build. Rejected; the `[lib.gpu]` profile is the whole
  point.
- **A separate `rosnet-gpu` crate/repo** — heavier (a new repo + release cadence + version matrix);
  the profile gives the same separation inside rosnet without a new crate.
- **The clean generic/transformer boundary** (only generic ops here, transformer ops at the consumer)
  — deferred per attn11 ADR 0017 as a premature public-API commitment; revisitable.
