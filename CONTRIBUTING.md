# Contributing to rosnet

## Development

1. Install the Cyrius toolchain at the version pinned in `cyrius.cyml`
   (`[package].cyrius`). The pin is the single source of truth — never hardcode
   a version elsewhere.
2. `cyrius deps` — resolve the `tyche` dep + stdlib into `lib/`.
3. `cyrius build src/main.cyr build/rosnet` — compile the smoke demo.
4. `cyrius test` — run the grad-check + BLAS-1 suite (`tests/rosnet.tcyr`).
5. `cyrius build tests/rosnet.bcyr build/bench && ./build/bench` — benchmarks.
6. `cyrius build tests/rosnet.fcyr build/fuzz && ./build/fuzz` — fuzz harness.
7. `cyrius distlib` — regenerate the consumable bundles (`dist/rosnet.cyr` from
   `[lib].modules`, `dist/rosnet-gpu.cyr` from `[lib.gpu].modules`). CI runs a
   drift gate, so regenerate and commit the bundle whenever lib surface changes.

There is no Makefile — the `cyrius` subcommands above are the loop.

See [`CLAUDE.md`](CLAUDE.md) for the full development loop and
[`docs/development/state.md`](docs/development/state.md) for current surface and
consumers.

## Gradient checks are the contract

rosnet ships hand-derived linear-algebra backward passes, and a hand-derived
backward is only as trustworthy as its grad-check. `linear_bwd` (`dx`/`dW`/`db`)
MUST be verified against **central finite differences** of `linear_fwd` in
`tests/rosnet.tcyr` before it lands — the same check ported from attn11, here as
rosnet's correctness gate. The bar is max relative error `< 1e-5` per gradient
array. The BLAS-1 ops (`t_sum`/`t_scale`/`t_axpy`/…) are checked exactly on
integer-valued inputs so bit patterns compare directly. A new op that touches a
backward path without a passing grad check is incomplete.

The GPU backend (`[lib.gpu]`, `dist/rosnet-gpu.cyr`) is validated against the
CPU path as the oracle — bit-exact / ~1e-13 tolerance — in the consumer
(attn11's GPU suite), since it requires mabda + a real AMD device.

## Numeric rules

- Cyrius has no float type — an `f64` is its bit pattern carried in an `i64`.
  Use the `f64_*` builtins (`f64_add`/`f64_mul`/…), never `+`/`*` on float
  values. `ff(n)` / `f64_from(n)` lift an integer literal to its f64 pattern.
- Build precise constants from integer ratios (e.g. `f64_div(f64_from(1),
  f64_from(10000))` for `1e-4`); long-digit float literals mis-parse.
- A tensor is a flat, row-major run of f64 in heap memory; shapes (`M`/`K`/`N`)
  live in the caller. Forward writes the output; backward writes input grads and
  accumulates into parameter grads. No allocation inside hot loops.
- Every buffer declaration is a contract: `var buf[N]` is N **bytes**, not N
  entries.

## Process

- One change at a time. Never bundle unrelated changes in a single PR.
- Test after every change; grad-check after every backward-touching change;
  benchmark after every perf-touching change.
- Performance claims must include numbers — `before → after` with the bench name.
- Breaking changes get a `Breaking` section in [`CHANGELOG.md`](CHANGELOG.md)
  with a migration paragraph.
- Do not commit/push or use `gh` — the maintainer handles git operations.

## License

GPL-3.0-only.
