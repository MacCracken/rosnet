# rosnet — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.2.0** — *added the GPU backend as a mabda-gated `[lib.gpu]` profile* (ADR 0001). `src/gpu.cyr`
extracted whole from attn11 (M18/M19, native-AMD f64 SPIR-V on mabda) → `dist/rosnet-gpu.cyr` (2323
lines): GPU foundation + generic tensor ops (matmul/ln/gelu/adam/linear_bwd/ln_bwd) + transformer ops
(attn_core/head/rope). **The CPU `[lib]` bundle (`dist/rosnet.cyr`) code is unchanged** (only the version
stamp moves to 0.2.0) — CPU-only consumers stay mabda-free; GPU consumers import `dist/rosnet-gpu.cyr` + declare `[deps.mabda]` (the
mabda symbols ship unresolved). Validated bit-exact / ~1e-13 vs the CPU oracle in the first consumer
(attn11 1.10.0, its 12-test GPU suite). cyrius pin unchanged (6.2.11); no top-level mabda dep.

(**0.1.1** — scaffolded 2026-06-11 via `cyrius init`.)

## Toolchain

- **Cyrius pin**: `6.2.11` (in `cyrius.cyml [package].cyrius`)

## Source

Initial scaffold only.

## Tests

- `tests/rosnet.tcyr` — primary suite (smoke + math; passes on `cyrius test`)
- `tests/rosnet.bcyr` — benchmark stub (no-op)
- `tests/rosnet.fcyr` — fuzz stub

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert

## Consumers

_None yet._

## Next

See [`roadmap.md`](roadmap.md).
