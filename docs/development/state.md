# rosnet — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**1.2.1 — 2026-11-09 (toolchain + dependency realign).** No source change:
cyrius `6.7.9` → `6.7.18` (the latest), tyche `1.2.0` → `1.2.2` (the latest),
vendored `lib/` re-synced to the 6.7.18 snapshot, `cyrius.lock` regenerated.
The 7,462-word golden dump of every CPU op is bit-identical to 1.2.0's, and so
is an 18,167-line dump of every GPU SPIR-V builder's output (checked under
6.6.20, 6.7.9 and 6.7.18); suite 23/23 unchanged.

(**1.2.0** 2026-11-09 — onto the cyrius 6.7 series: cyrius → 6.7.9, tyche →
1.2.0, tests run as `cyrius test tests`. **1.1.3** 2026-11-09 — cyrius → 6.6.20, tyche → 1.1.1; smaller sidecars; the
GPU bundle's reliance on the consumer's `puts` documented. **1.1.2** 2026-09-11 — cyrius → 6.6.2; fixed the GPU bundle's retired
`payload` accessor. **1.1.1** 2026-08-17 — cyrius → 6.5.27, tyche → 1.0.1.)

**1.1.0 — RELEASED 2026-07-05 (conv2d/conv1d).** The 1.1 additive: NCHW,
per-axis stride/pad, hand-derived FD-gated gradients (incl. strided/padded/
non-square configs), conv1d = exact H=1 wrappers; suite 7→**23**; dist
245→390 lines; benched ~1.1 G MAC/s. **The modality axis is now
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
  release. **Bumped 6.7.9 → 6.7.18 on 2026-11-09.** 23/23 before and after,
  every CPU op's output bit-identical, and an interleaved A/B benchmark within
  1%. `lib/` byte-matches the 6.7.18 snapshot (113 files).
- **Bump procedure:** edit the pin → `cyrius lib sync --full` (cyrius refuses a
  moved pin until this runs) → `cyrius deps` → if a git dep's tag moved, delete
  `cyrius.lock` and run `cyrius deps` again (the lock keeps the old pin
  fail-closed) → `cyrius distlib` + `cyrius distlib gpu`. Commit the pin,
  `lib/` and `cyrius.lock` together, or CI's `cyrius deps` fails.
- `lib/agnosys.cyr` is a leftover from the 2026-06-11 scaffold. It is in no
  6.6.x snapshot and nothing in `src/` or `tests/` uses it, but `lib sync` does
  not prune it, so it stays in `lib/` and in `cyrius.lock`.
- Earlier: 6.2.11 → 6.5.27 (2026-08-17, ended the deliberate deep-lag),
  6.5.27 → 6.6.2 (2026-09-11), 6.6.2 → 6.6.20 and 6.6.20 → 6.7.9
  (2026-11-09).

## Source

- `src/tensor.cyr` — storage + BLAS-1 (`t_alloc`/`t_zero`/`t_fill`/`t_copy`,
  `tget`/`tset`/`ff`, `t_axpy`/`t_add_into`/`t_scale`/`t_sum`, `f64_is_finite`,
  `t_randn`).
- `src/linear.cyr` — dense matmul + gradient (`linear_fwd`/`linear_bwd`).
- `src/conv.cyr` — conv2d/conv1d fwd+bwd (1.1 additive; per-axis stride/pad).
- `src/gpu.cyr` — GPU backend (`[lib.gpu]` profile, ADR 0001; mabda-gated,
  Linux-only), extracted whole from attn11 (M18/M19).
- `src/main.cyr` — smoke demo (`main()` + `SYS_EXIT`; excluded from the bundle).

CPU `[lib]` bundle → `dist/rosnet.cyr` (390 lines, v1.2.1; sidecar
`dist/rosnet.deps` = alloc, string, fmt, syscalls); GPU profile →
`dist/rosnet-gpu.cyr` (2323 lines, v1.2.1; sidecar = mabda, math, result,
ganita). Regenerate with `cyrius distlib` + `cyrius distlib gpu`; verify both
with `cyrius distlib --all --check`. The GPU bundle calls `puts`, which no
stdlib leaf defines — its consumer must supply it (attn11 does).

## Tests / benches

- `tests/rosnet.tcyr` — primary suite: exact BLAS-1 checks, central
  finite-difference grad-checks of `linear_bwd` (dx/dW/db, rel err < 1e-5),
  conv known-answer + grad-checks, conv1d wrapper parity.
  **23/23 passing** on `cyrius test tests` (at cyrius 6.7.18).
- `tests/rosnet.bcyr` — the v1.0 benchmark harness (consumer-shaped sizes);
  numbers in [`docs/benchmarks.md`](../benchmarks.md).
- `tests/rosnet.fcyr` — fuzz stub (no untrusted-data parsing on this surface —
  see the audit; becomes real if a quantized-dtype loader ever lands).

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

Post-1.0 additive lane (demand-gated): conv2d fwd+bwd (the modality/Type-4
gate), the f32/f16/bf16 wideners promotion, CPU matmul optimization when a
consumer is throughput-blocked. See [`roadmap.md`](roadmap.md).
