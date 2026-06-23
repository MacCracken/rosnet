# Security Policy

## Reporting

Report vulnerabilities to **cyriusmaccken@gmail.com**. Include reproduction
steps and the rosnet version from `VERSION` (currently **0.2.0**). Expect an
initial response within one week. Coordinated disclosure is appreciated — do not
open a public GitHub issue with exploit details.

## Threat model

rosnet is a **single-process, in-library** dense f64 tensor algebra layer
(storage, BLAS-1, matmul + its gradient). It has **no networking**, opens **no
files**, parses **no formats**, and spawns **no subprocesses**. It is linked
into a host program and called directly.

Its only untrusted input is what the **calling program** passes: tensor element
data and the shape/length arguments (`n`, `M`/`K`/`N`) that index into it. A
tensor is a raw heap pointer to a flat row-major run of f64; rosnet trusts the
caller to pass a buffer that matches the dimensions it also passes. There is no
bounds metadata carried with a tensor, so:

- Mismatched dimensions (a buffer smaller than `M*K`, etc.) cause out-of-bounds
  reads/writes — the caller owns dimension validation. rosnet does **not** clamp
  or re-derive shapes.
- `t_alloc(n)` allocates `n` f64 from the host allocator; a hostile-huge `n`
  fails the allocation cleanly rather than wrapping. There is no per-call size
  cap — the bound is the host's allocator.
- f64 data is treated as opaque IEEE-754 bit patterns; `f64_is_finite` is
  available for callers that want to reject NaN/Inf, but rosnet does not
  validate the *values* it computes over.

rosnet *does not* defend against:

- a caller that already controls the host process (it shares that address space
  and is equivalent to code execution),
- remote / network attacks (rosnet has no networking),
- side channels (timing, cache),
- adversarial dimension/data passed by a buggy or hostile caller — validating
  shapes against buffers is the caller's responsibility, as documented above.

### GPU backend (`[lib.gpu]`, Linux-only)

The optional GPU profile (`dist/rosnet-gpu.cyr`) layers on **mabda**'s native
AMD f64 SPIR-V → GFX9 device/dispatch primitives and requires a real DRM device.
It hand-emits SPIR-V and dispatches compute over a fixed GTT buffer pool; the
same caller-trusts-its-own-dimensions model applies, now against device memory.
This surface is **Linux/DRM-only** and absent from the default CPU bundle and the
AGNOS build — a CPU-only consumer never links it. It is validated for
correctness (bit-exact / ~1e-13 vs the CPU oracle in the attn11 consumer), not
yet hardened against a hostile caller.

### PRNG

`t_randn` draws from **tyche**'s deterministic statistical PRNG
(`rng_seed`/`rng_normal`), used for weight initialization only. It is **not**
cryptographic — never use it for any security purpose (that is sigil's job).

## Maturity

rosnet is **pre-1.0 (0.2.0)** and has **not** had a formal security audit. The
threat model above describes the surface honestly; it does not claim mitigations
that do not exist. A `docs/audit/YYYY-MM-DD-audit.md` pass is a gate on the v1.0
criteria (see [`docs/development/roadmap.md`](docs/development/roadmap.md)) and
this document will be expanded when that audit lands.

The Cyrius toolchain version is pinned in `cyrius.cyml` (single source of truth),
and CI installs it via the upstream `install.sh` so the pin is enforced.
