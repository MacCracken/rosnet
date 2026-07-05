# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.0.0] - 2026-07-04

**v1.0 — the substrate freeze.** No behavior change from 0.2.0: the cut freezes
what **six shipping consumers** (attn11, tarka, tentib, prajna, rupantara,
anukūlana) have exercised unchanged since 0.1.1 — the strongest dogfooding
signal in the ecosystem — and closes the remaining v1.0 criteria. The whole
surface rides the freeze, **CPU bundle + GPU profile** (mabda 4.x churn is
provider-level — 4.0 added basic NVIDIA support — not the device/dispatch API
the profile layers on).

### Added
- **`docs/api.md`** — the frozen 1.x public surface: CPU (`t_*` storage/BLAS-1,
  `linear_fwd`/`linear_bwd`, guards/init) + GPU (gates `g_gpu`/`g_gpu_tc`,
  13 ops in bit-exact/tolerance classes, `gpu_probe` + 11 engagement counters),
  and the **memory contract** (flat row-major f64, shapes live in the caller,
  no bounds checks by design). `_`-prefixed internals + `GPU_TK`/`GPU_BO_MAX`
  tuning + the smoke `main()` stay out-of-freeze.
- **`docs/benchmarks.md`** — v1.0 capture (Ryzen 7 5800H, cycc 6.4.2):
  `t_axpy` n=768 **2 µs** · `linear_fwd` 64×768×768 **30.9 ms** ·
  `linear_fwd` 1×768×12564 (¼ LM-head row) **8.5 ms** (full head ≈ 34 ms) ·
  `linear_bwd` 64×768×768 **73.5 ms** (≈2.4× fwd). Read honestly: these are
  correctness-first reference kernels ~1–2 orders behind a tuned BLAS — the
  known cost of the sovereign substrate; closing it is post-1.0 (or GPU) work.
- **`tests/rosnet.bcyr`** — the real benchmark harness replacing the no-op stub
  (consumer-shaped sizes: GPT-2 projection, LM-head row, backward pair).
- **`docs/audit/2026-07-04-audit.md`** — v1.0 hardening audit: **PASS, 0 code
  changes**. The no-bounds-checks posture is audited as a deliberate substrate
  contract (hot-path caller-guarantees, upheld by construction in all six
  consumers), NOT tarka-style `guard()`s; `t_alloc` fail-loud + GPU
  guard-before-upload (dims / dispatch-dim / BO-fit on every device entry) are
  the standing 1.x invariants.

### Changed
- README: GPU-profile section + docs pointers + dep tag → 1.0.0; CLAUDE.md
  identity corrected (Library, not Binary; mission statement filled in).

### v1.0 criteria — all met
✅ API frozen + documented (`docs/api.md`) · ✅ tests (7/7 incl. central-FD
grad-checks of `linear_bwd`; GPU validated in-consumer vs the CPU oracle) ·
✅ benchmarks (`docs/benchmarks.md`) · ✅ downstream consumers green (six) ·
✅ CHANGELOG complete from 0.1.1 · ✅ security audit
(`docs/audit/2026-07-04-audit.md`).

## [0.2.0] - 2026-06-21

### Added
- **GPU backend** as a separate **`[lib.gpu]` profile** → `dist/rosnet-gpu.cyr` (ADR 0001).
  Extracted whole from attn11 (M18/M19), it is the GPU sibling of rosnet's SIMD path, layered on
  **mabda**'s native-AMD f64 SPIR-V→GFX9 device/dispatch primitives: the GPU foundation (device/ctx,
  a 7-buffer GTT pool, hand-emitted SPIR-V primitives, tiled-matmul builders, a shader cache,
  dispatch) + the generic tensor ops (`gpu_matmul_fwd`, `gpu_ln_fwd`, `gpu_gelu_fwd`/`bwd`,
  `gpu_linear_bwd`, `gpu_ln_bwd`, `gpu_adam_step`) + the transformer ops (`gpu_attn_core`/`bwd`,
  `gpu_head_fwd`/`bwd`, `gpu_rope_apply`). Validated bit-exact / ~1e-13-tolerance vs the CPU oracle
  in the consumer (attn11's 12-test GPU suite).
- `src/gpu.cyr` (the backend) + the `[lib.gpu]` profile in `cyrius.cyml`.

### Notes
- **The default CPU `[lib]` bundle (`dist/rosnet.cyr`) is unchanged** (only its version stamp moves
  `0.1.1`→`0.2.0`; the code is byte-identical) — the GPU backend is a *separate* profile, so CPU-only
  consumers stay **mabda-free** (no Linux-only surface, no size bloat). `dist/rosnet-gpu.cyr` leaves mabda's device/dispatch symbols **unresolved** (the
  consumer-included-bundle pattern); a GPU consumer imports it **and** declares `[deps.mabda]`
  (Linux-only). rosnet declares **no** top-level mabda dep (cyrius auto-prepends a dep's modules on
  every target → would contaminate the CPU bundle + the AGNOS build).
- No CPU-path change; consumers on `dist/rosnet.cyr` need no migration.

## [0.1.1]

### Changed
- Bumped Cyrius toolchain pin to `6.2.11` (`cyrius.cyml [package].cyrius`).
- Bumped `tyche` dependency to `0.1.1`.

## [0.1.0]

### Added
- Initial project scaffold.
- Tensor storage + BLAS-1 layer (`src/tensor.cyr`), extracted from attn11's
  tensor layer at its v1 freeze: `t_alloc`/`t_zero`/`t_fill`/`t_copy`,
  `tget`/`tset`/`ff`, `t_axpy`/`t_add_into`/`t_scale`/`t_sum`, `f64_is_finite`,
  and `t_randn` (normal-fill via the tyche dependency).
- Dense matmul + gradient (`src/linear.cyr`): `linear_fwd`
  (`y = x·W + b`, 4-wide SIMD) and `linear_bwd` (`dx = dy·Wᵀ`, `dW = xᵀ·dy`).
  Matmul and its backward are pure linear algebra, so both live here rather
  than in a neural-net layer.
- `tyche` dependency (`[deps.tyche]`, 0.1.0) — supplies `rng_seed`/`rng_normal`
  for `t_randn`. The only transitive dep; pure tensor/matmul ops need none.
- Test suite (`tests/rosnet.tcyr`): exact integer-valued BLAS-1 checks +
  **central finite-difference grad-checks** of `linear_bwd` (dx/dW/db, rel err
  < 1e-5), ported from attn11. 7/7 passing.
- `dist/rosnet.cyr` consumable bundle via `cyrius distlib` (leaves tyche's
  `rng_normal` unresolved by design — supplied by the consumer's tyche dep).
- CI/release workflows: upstream `install.sh` toolchain install (passes the
  `cyrius deps` pin-check), `workflow_call` CI gate, fmt/lint, build+ELF+smoke,
  tests, bench, fuzz, distlib drift gate, and a tag-driven release shipping the
  source tarball + version-stamped `dist/rosnet.cyr` + SHA256SUMS. Modeled on
  patra/sigil.
