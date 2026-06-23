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

- `src/tensor.cyr` — storage + BLAS-1 (`t_alloc`/`t_zero`/`t_fill`/`t_copy`,
  `tget`/`tset`/`ff`, `t_axpy`/`t_add_into`/`t_scale`/`t_sum`, `f64_is_finite`,
  `t_randn`).
- `src/linear.cyr` — dense matmul + gradient (`linear_fwd`/`linear_bwd`).
- `src/gpu.cyr` — GPU backend (`[lib.gpu]` profile, ADR 0001; mabda-gated,
  Linux-only), extracted whole from attn11 (M18/M19).
- `src/main.cyr` — smoke demo (`main()` + `SYS_EXIT`; excluded from the bundle).

CPU `[lib]` bundle → `dist/rosnet.cyr`; GPU profile → `dist/rosnet-gpu.cyr`.

## Tests

- `tests/rosnet.tcyr` — primary suite: exact BLAS-1 checks + central
  finite-difference grad-checks of `linear_bwd` (dx/dW/db, rel err < 1e-5).
  **7/7 passing** on `cyrius test`.
- `tests/rosnet.bcyr` — benchmark stub (no-op timing harness).
- `tests/rosnet.fcyr` — fuzz stub.

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math
- **tyche** (`0.1.1`) — deterministic statistical PRNG; supplies
  `rng_seed`/`rng_normal` for `t_randn`. The only non-stdlib dep; pure
  tensor/matmul ops need none. (The GPU profile additionally needs **mabda**,
  supplied by the consumer — rosnet declares no top-level mabda dep.)

## Consumers

- **attn11** (1.10.0+) — consumes the GPU backend (`dist/rosnet-gpu.cyr`),
  validated against its CPU path as the oracle. Also targeted by **tarka** and
  **tentib** on the same f64 substrate.

## Next

See [`roadmap.md`](roadmap.md).
