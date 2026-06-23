# rosnet — Roadmap

> Milestone plan through v1.0. State lives in [`state.md`](state.md);
> this file is the sequencing — what ships, in what order, against
> what dependency gates.

## v1.0 criteria

- [ ] Public API frozen — every exported symbol documented and tested
- [ ] Test coverage adequate for the surface area
- [ ] Benchmarks captured in `docs/benchmarks.md`
- [ ] At least one downstream consumer green
- [ ] CHANGELOG complete from v0.1.0 onward
- [ ] Security audit pass (`docs/audit/YYYY-MM-DD-audit.md`)

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

### M2 — _Title_ (v0.3.0)

_Replace with the next real milestone. Specify the user-visible change, the dep gates, and the acceptance criteria._

## Out of scope (for v1.0)

_Capture what's deliberately NOT in scope for v1.0. The list keeps future contributors from adding to v1.0 by accident._

- _e.g. Windows support, GUI front-end, etc._
