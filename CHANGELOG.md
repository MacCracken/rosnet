# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
