# 0002 — Kernel optimizations preserve every output's rounding sequence

**Status**: Accepted
**Date**: 2026-11-09

## Context

rosnet's kernels are the oracle other things are checked against. attn11's GPU
suite compares its device results with these CPU ops bit for bit (the
bit-exact class) or within ~1e-13 (the tolerance class); consumers' training
runs are byte-reproducible ("a byte-identical training run proves the swap",
attn11's manifest); and the golden dumps in this repo's audits hash raw f64 bit
patterns. Floating-point addition is not associative, so any change to the
order in which an output's terms are added — a wider SIMD reduction, a
reordered loop nest, a fused multiply-add — changes the low bits of the result.

The 1.2.2 P(-1) sweep had large speed-ups available ([`benchmarks.md`](../benchmarks.md)).
Some keep every output's sequence of roundings: one packed AXPY per row instead
of one per four elements, vectorizing conv along the output row, hoisting the
padding-window bounds tests. Others do not: an 8-lane dx reduction, summing
dW's products before adding them, `f64v_axpy` for `t_axpy` (aarch64 fuses it).
Those are different kinds of change and need a rule.

## Decision

**A kernel optimization must leave every output with the same sequence of
roundings on every target it already matched.** Concretely:

- each output's terms are added in the same order, starting from the same
  value (bias, 0, or the accumulated gradient);
- each product and sum is rounded the same number of times — a packed op may
  replace a scalar op only if it rounds identically on that target (on x86_64
  `mulpd`/`addpd`, `f64v_scale`/`f64v_add`/`f64v_mul` everywhere; not a packed
  multiply-add on aarch64, which is a fused `FMLA`);
- the vector/scalar split of an already-vectorized kernel stays where it was
  (`linear_fwd`'s `n4`), so no element changes path on a target where the two
  paths round differently.

The gate is a golden dump of raw bit patterns over every CPU op (odd tails,
chunk edges, padding windows that miss the input, K = 0, no-bias paths),
byte-identical before and after, plus the plain-loop oracles in
`tests/reference.cyr` checked by the suite and the fuzz harness on every CI run.
The one stated exception: a NaN produced from two NaN operands may carry
either operand's payload.

A change that alters bits is not an optimization under this rule. It is a
numerical change: its own minor release, its own CHANGELOG entry saying which
outputs move and by how much, and a heads-up to the consumers whose suites
compare bit patterns.

## Consequences

- **Positive** — consumers can take any 1.x patch release without re-baselining
  golden outputs or GPU parity tests; an optimization's correctness reduces to
  a byte comparison, which a mutation check showed catches order changes (kw
  walked the wrong way, dW pre-summed), dropped tails and chunk-edge losses.
- **Negative** — the largest remaining wins are off the table for a patch:
  AVX2 or wider dx lanes, blocked/tiled matmul with a different k-order,
  threads with per-thread partial sums, and `t_axpy` on the packed path.
  `t_axpy` stays scalar specifically so it does not pick up the aarch64
  FMA divergence that `linear_fwd` already has.
- **Neutral** — the oracles in `tests/reference.cyr` are now part of the
  contract and must change in the same commit as any deliberate numerical
  change.

## Alternatives considered

- **Tolerance-gated optimizations** (accept any change within ~1e-12): rejected.
  It silently breaks every consumer that compares bits (attn11's bit-exact GPU
  class, byte-reproducible training) and makes "did this optimization change
  anything?" unanswerable.
- **Per-target bit-identity only on x86_64** (let aarch64 drift further): rejected
  for new code. aarch64 already differs in `linear_fwd` / `linear_bwd` because
  cyrius fuses the packed multiply-add there; this ADR keeps that set from
  growing (hence `t_axpy` scalar) rather than widening it.
