# Security Policy

## Reporting

Report vulnerabilities to **cyriusmaccken@gmail.com**. Include reproduction
steps and the rosnet version from `VERSION`. Expect an
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
- `t_alloc(n)` allocates `n` f64 from the host allocator and never returns a
  null tensor: a negative `n`, an `n` whose byte size `8n` would overflow, or an
  allocation the allocator refuses prints to stderr and exits. (Before 1.2.2 an
  `n` just above 2^61 wrapped `8n` to a small size, the allocation succeeded,
  and the zero-fill ran off the block — see `docs/audit/2026-11-09-audit.md`.)
  The bound is the host allocator's per-call cap.
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
same caller-trusts-its-own-dimensions model applies to the caller's own
buffers. What the GPU path adds is host-side state of its own — the fixed GTT
pool, the shader cache, the SPIR-V build buffers — and every entry point guards
those before touching them: each dim positive, every upload within its 4 MB
buffer object, every grid within mabda's dispatch cap, every shader-cache key
field in range, and the untiled attention builders capped at head dim 32 so a
shader fits its build buffer (before 1.2.2 the forward attention path had no
such cap and overran a stack buffer at head dim 64). Anything outside those
bounds falls back to the CPU op, leaving the caller's outputs untouched (since
1.3.0), and a failing dispatch disables the GPU path for the rest of the
process. This surface is **Linux/DRM-only** and absent
from the default CPU bundle and the AGNOS build — a CPU-only consumer never
links it. It is validated for correctness (bit-exact / ~1e-13 vs the CPU oracle
in the attn11 consumer); its device trust boundary is mabda's.

### PRNG

`t_randn` draws from **tyche**'s deterministic statistical PRNG
(`rng_seed`/`rng_normal`), used for weight initialization only. It is **not**
cryptographic — never use it for any security purpose (that is sigil's job).

## Maturity

rosnet's 1.x surface is frozen ([`docs/api.md`](docs/api.md)). Audits:
[`docs/audit/2026-07-04-audit.md`](docs/audit/2026-07-04-audit.md) (the v1.0
gate) and [`docs/audit/2026-11-09-audit.md`](docs/audit/2026-11-09-audit.md)
(the 1.2.2 P(-1) hardening sweep). The threat model above describes the surface
honestly; it does not claim mitigations that do not exist.

The Cyrius toolchain version is pinned in `cyrius.cyml` (single source of truth),
and CI installs it via the upstream `install.sh` so the pin is enforced.
