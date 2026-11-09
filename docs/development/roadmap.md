# rosnet — Roadmap

> Milestone plan through v1.0. State lives in [`state.md`](state.md);
> this file is the sequencing — what ships, in what order, against
> what dependency gates.

## v1.0 criteria — ✅ ALL MET (1.0.0 cut 2026-07-04)

- [x] Public API frozen — every exported symbol documented and tested (`docs/api.md`)
- [x] Test coverage adequate for the surface area (7/7 incl. central-FD grad-checks; GPU validated in-consumer vs the CPU oracle)
- [x] Benchmarks captured in `docs/benchmarks.md` (v1.0 capture, Ryzen 7 5800H)
- [x] At least one downstream consumer green — **six** (attn11, tarka, tentib, prajna, rupantara, anukūlana)
- [x] CHANGELOG complete from v0.1.0 onward
- [x] Security audit pass (`docs/audit/2026-07-04-audit.md` — PASS, 0 code changes; the no-bounds-checks substrate contract audited as deliberate)

## Milestones

### M0 — Scaffold (v0.1.0) — ✅ shipped 2026-06-11

- `cyrius init` scaffold landed
- Doc-tree per [first-party-documentation.md](https://github.com/MacCracken/agnosticos/blob/main/docs/development/applications/first-party-documentation.md)
- ADRs / architecture notes / guides / examples folders ready

### M1 — GPU backend (v0.2.0) — ✅ shipped 2026-06-21

- `src/gpu.cyr` added as a separate `[lib.gpu]` profile → `dist/rosnet-gpu.cyr`
  (ADR 0001), extracted whole from attn11 (M18/M19): native-AMD f64 SPIR-V on
  mabda — GPU foundation + generic tensor ops + transformer ops.
- CPU `[lib]` bundle left byte-identical (mabda-free); GPU consumers import the
  GPU bundle and declare `[deps.mabda]` (Linux-only).
- Validated bit-exact / ~1e-13 vs the CPU oracle in attn11 (1.10.0, GPU suite).

### M2 — v1.0 freeze (v1.0.0) — ✅ shipped 2026-07-04

- Public surface frozen (`docs/api.md`): CPU bundle + GPU profile both ride the
  freeze — mabda 4.x churn is provider-level (4.0 added basic NVIDIA), not the
  device/dispatch API the profile layers on.
- Benchmarks captured (`docs/benchmarks.md`), audit passed (`docs/audit/`),
  no behavior change from 0.2.0 (a tarka-style clean cut).

## Post-1.0 (additive 1.x minors, demand-gated)

- **conv2d / conv1d fwd+bwd — ✅ LANDED 2026-07-05** (**cut as 1.1.0**, 2026-07-05): NCHW, per-axis stride/pad, hand-derived dx/dW/db, FD-gated
  across strided/padded/non-square configs + conv1d bit-parity; suite 7→23;
  benched. Opens the modality axis (sight proof next) AND Type-4 diffusion.
- **IEEE-754 wideners** (f32/f16/bf16→f64) — promotion candidate from
  anukūlana's importer on a second consumer (GGUF parser / another importer).
- **CPU kernel optimization** — 1.2.2 (P(-1) sweep) took the bit-identical
  wins: `linear_fwd` / dW as one packed AXPY per row (2.2–2.7×), stride-1 conv
  along the output row (fwd 8–13×, bwd 3–3.6×), packed `t_scale` /
  `t_add_into`. What remains changes bits or the bundle's leaves, so it waits
  for a consumer that is throughput-blocked: blocking/tiling, threads, AVX2
  (`f64v256_*` behind a runtime CPU check), a wider dx reduction, im2col for
  strided conv.
- **Quantized dtypes** — steps outside the caller-guarantees contract; must
  bounds-check like tula/anukūlana parsers (see the audit's standing guidance).

## Out of scope (for 1.x)

- Bounds/shape checking on the hot path — the audited substrate contract is
  caller-guarantees; revisiting it is a 2.0 question.
- A murti-style resident-model/runtime layer (hoosh's seam, per
  `agnosticos docs/development/planning/murti.md`).
- GPU provider work (mabda's surface, not rosnet's).
