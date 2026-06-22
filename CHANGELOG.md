# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
