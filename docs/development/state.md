# rosnet — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

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

- **Cyrius pin**: `6.2.11` (in `cyrius.cyml [package].cyrius`). Deliberately
  deep-lag: the CPU surface is 245 bundled lines of stable arithmetic; the
  6.3.13 array-locals-to-stack change is safe here (the SIMD accumulators are
  32-byte locals). Bump opportunistically with a consumer-driven touch.

## Source

- `src/tensor.cyr` — storage + BLAS-1 (`t_alloc`/`t_zero`/`t_fill`/`t_copy`,
  `tget`/`tset`/`ff`, `t_axpy`/`t_add_into`/`t_scale`/`t_sum`, `f64_is_finite`,
  `t_randn`).
- `src/linear.cyr` — dense matmul + gradient (`linear_fwd`/`linear_bwd`).
- `src/gpu.cyr` — GPU backend (`[lib.gpu]` profile, ADR 0001; mabda-gated,
  Linux-only), extracted whole from attn11 (M18/M19).
- `src/main.cyr` — smoke demo (`main()` + `SYS_EXIT`; excluded from the bundle).

CPU `[lib]` bundle → `dist/rosnet.cyr` (245 lines, v1.0.0); GPU profile →
`dist/rosnet-gpu.cyr` (2323 lines, v1.0.0; regenerate with
`cyrius distlib` + `cyrius distlib gpu`).

## Tests / benches

- `tests/rosnet.tcyr` — primary suite: exact BLAS-1 checks + central
  finite-difference grad-checks of `linear_bwd` (dx/dW/db, rel err < 1e-5).
  **7/7 passing** on `cyrius test`.
- `tests/rosnet.bcyr` — the v1.0 benchmark harness (consumer-shaped sizes);
  numbers in [`docs/benchmarks.md`](../benchmarks.md).
- `tests/rosnet.fcyr` — fuzz stub (no untrusted-data parsing on this surface —
  see the audit; becomes real if a quantized-dtype loader ever lands).

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math
- **tyche** (`0.1.1`) — deterministic statistical PRNG; supplies
  `rng_seed`/`rng_normal` for `t_randn`. The only non-stdlib dep; pure
  tensor/matmul ops need none. (The GPU profile additionally needs **mabda**,
  supplied by the consumer — rosnet declares no top-level mabda dep.)

## Consumers (six, all green at the freeze)

**attn11** (GPU backend consumer + oracle suite) · **tarka** · **tentib** ·
**prajna** · **rupantara** (the transformer-forward lib) · **anukūlana**
(Type-3 import — the real-GPT-2 forward runs on this substrate).

## Next

Post-1.0 additive lane (demand-gated): conv2d fwd+bwd (the modality/Type-4
gate), the f32/f16/bf16 wideners promotion, CPU matmul optimization when a
consumer is throughput-blocked. See [`roadmap.md`](roadmap.md).
