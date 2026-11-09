# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.3.0] - 2026-11-09

Closes the 1.2.2 sweep's open ends ([`roadmap.md`](docs/development/roadmap.md)).
No API change. Every CPU result and every generated GPU shader is byte-identical
to 1.2.2 (the 15,302-line CPU golden dump and the 18,167-line SPIR-V builder
dump).

### Validated on a device

- **The 1.2.2 GPU fixes, checked on hardware.** They had been verified on the
  host only. On an AMD GFX9 GPU (Radeon Vega, Cezanne), attn11's 12-test GPU
  suite passes against this release — six ops bit-exact to the CPU, six within
  their tolerances. It was built from a scratch copy at cyrius 6.7.18; attn11
  itself is unchanged.
- **The attention-cache fix, reproduced.** With 1.2.1's code, `gpu_attn_core`
  at head dim 8, called after head dim 4 with the same C and T, returns 0 and a
  wrong `concat`. With 1.3.0 both match the CPU reference.

### Changed

- **A GPU fallback leaves the caller's outputs untouched.** Five entries wrote
  an output, then a later stage could still fall back: `gpu_ln_fwd` (mean /
  rstd), `gpu_linear_bwd` (dx), `gpu_head_bwd` (D_f), `gpu_ln_bwd` (dx) and
  `gpu_attn_core_bwd` (dQ, dV). Each stage's result now stays in a device
  buffer of its own, and every output is read back after the last dispatch.
  `docs/api.md` restores the 1.0 wording.
- **A full shader cache no longer sends ops to the CPU.** Past 1024 cached
  kernels, a new kernel is compiled, dispatched once and released. Dispatch is
  synchronous, and every call site dispatches the kernel it fetched exactly
  once. 1.2.1 leaked a module per call here, and 1.2.2 fell back to the CPU.
- **A failing dispatch is sticky.** Every shape is guarded before it reaches
  the device, so a dispatch error means the device or driver is in trouble.
  Every later GPU call now falls back at once instead of re-uploading up to
  12 MB first. All 20 dispatch sites go through one wrapper, `_gpu_dispatch`.
- Fewer host copies: `gpu_linear_bwd` uploads dy once, not twice, and
  `gpu_head_bwd` uploads D_logits once, not twice.
- Cache keys no longer include values their kernels do not bake in: C and hd
  for rowmax, expsum and dot_t (their builders never take them), and M for
  lnbwd_param (its SPIR-V is byte-identical across M, checked). Those shapes
  now share kernels.

### Added

- **`tests/gpu_device.tcyr`**, 39 assertions run on the device. It covers
  bit-exact matmul, the attention cache-key pair, the guard fallbacks, the soft
  cache cap, and every multi-stage entry's outputs staying untouched when its
  last stage falls back. To force that, the test marks the last stage's kernel
  as a known compile failure, and checks that the earlier stages ran. Without a
  device the file prints SKIP and exits 0 (checked with `/dev/dri` hidden). It
  catches 1.2.1's attention code and a moved-back `dx` readback.
- `tests/gpu_host.tcyr` 60 → 71 assertions: transient module lifetime, and the
  sticky dispatch failure.
- **An aarch64 CI leg** under qemu-user, same shape as tyche's. It runs the CPU
  suite (60 assertions on aarch64, where the linear cases take the tolerance
  branch), the GPU host tests and the fuzz harness. This is the first time
  rosnet's per-target numerical claims actually run on aarch64; all hold,
  including the conv and BLAS-1 kernels matching the plain loops bit for bit.

### Documentation

- **Correction: GPU attention's head-dim limits.** The 1.2.2 audit estimated
  that forward attention compiles up to hd ≈ 26, from the SPIR-V id bound
  alone. mabda's compiler also needs ids for the values it synthesizes. On the
  device at cyrius 6.7.18, the forward scores kernel compiles only up to
  **hd 11**, and the backward dp kernel up to **hd 17**. Corrected in the audit,
  `docs/api.md`, the roadmap and `src/gpu.cyr`.

## [1.2.2] - 2026-11-09

The P(-1) hardening, optimization and repair sweep, recorded in
[`docs/audit/2026-11-09-audit.md`](docs/audit/2026-11-09-audit.md). No public
signature changed, and every CPU result is bit-identical to 1.2.1: a
15,302-line golden dump of raw f64 bit patterns over every op is byte-identical,
and so is an 18,167-line dump of every GPU shader builder's output.

### Fixed

- **A stack-buffer overflow in `gpu_attn_core`** (HIGH). It had no head-dim
  cap, and the scores shader builder unrolls the head dim into a 8,192-byte
  stack buffer: at hd = 64 (GPT-2's head size) it wrote 11,172 bytes, before
  mabda's 256-id cap could refuse the shader. hd > 32 now falls back to the
  CPU, as the backward entry already did.
- **`gpu_attn_core` could return a wrong result with rc 0.** Its shader-cache
  key left out the head dim, which the kernels bake in, so two head counts
  sharing C and T reused each other's kernel. The key now includes it, and both
  attention entries require `C % nh == 0` — the backward assumed it and,
  without it, wrote past dK / dV.
- **A full shader cache leaked a compiled module on every call** and recompiled
  every time. It stopped at 128 entries though its arrays hold 1024. Capacity
  is now 1024, and a miss on a full cache falls back to the CPU without
  compiling.
- **`t_alloc(0)` exited the process** ("out of memory, requested f64s: 0"). It
  now returns a valid, distinct pointer — an empty tensor. A negative `n`, or
  one so large that `8n` wraps (it used to wrap to a small block and the
  zero-fill ran off the end), exits with "invalid size". Errors now go to
  stderr.
- GPU guards checked products, not dims: `gpu_matmul_fwd(M = −1, N = −1, …)`
  passed. Every entry now checks each dim; `gpu_ln_bwd` also checks M > 0 and
  that gamma fits its buffer; `gpu_adam_step(NP ≤ 0)` falls back instead of
  counting a dispatch that never ran.
- Shader-cache keys could overflow one field into the next for dims ≥ 16,384
  (degenerate shapes, but silent: a kernel ran for the wrong tile). Every key
  field is range-checked; out of range falls back to the CPU.
- **The GPU bundle no longer needs the consumer to define `puts`.** `gpu_probe`
  called a `puts` that no stdlib module defines; it linked only because attn11
  defines its own. It now writes the same bytes itself. The GPU sidecar gains
  `syscalls` and `string`; every consumer already declares both.
- The attention tile width is a constant, not a mutable global (16 would
  overflow the build buffer; 0 divides by zero).

### Changed — faster, same bits

Same session, median of three runs ([`docs/benchmarks.md`](docs/benchmarks.md)):

- **`linear_fwd`** 33.4 → 12.4 ms (64×768×768), 9.07 → 4.08 ms (LM-head row):
  one packed `f64v_axpy` per row step instead of one `f64v_fmadd` per four
  elements after filling a broadcast buffer.
- **`linear_bwd`** 70.4 → 43.3 ms: the same for dW.
- **conv**: `conv2d_fwd` 33.4 → 4.15 ms, `conv2d_bwd` 45.2 → 15.2 ms,
  `conv1d_fwd` 48.5 → 3.64 ms, `conv1d_bwd` 66.4 → 18.7 ms. The padding window
  is computed once per output instead of tested per multiply-add, and stride-1
  convs run along the output row with packed ops.
- **`t_scale`** 1,707 → 192 ns, **`t_add_into`** 1,729 → 254 ns (n = 768).

Every change keeps each output's sequence of roundings, which
[ADR 0002](docs/adr/0002-kernel-optimizations-preserve-bits.md) makes the rule
for optimizations. `t_axpy`, `t_sum` and `linear_bwd`'s dx are unchanged on
purpose.

### Changed

- **Tests: 23 → 114 assertions.** `tests/reference.cyr` holds the plain-loop
  forms of the kernels (the 1.2.1 conv loops verbatim); the suite checks every
  optimized kernel against them, covers the seven public functions that had no
  test, and `tests/gpu_host.tcyr` checks the GPU profile's host side — cache,
  key fields, every shader builder fitting its buffer — with no device.
- **The fuzz harness is real**: seeded property fuzzing, 15,000 random shapes
  per run with ±0, subnormal, huge and infinite inputs, against the same oracles.
- **CI**: lint fails on any warning in `src/` and `tests/`; a `vet` step; the
  drift check regenerates both bundles (a stale GPU bundle used to pass);
  read-only token by default; actions pinned by commit SHA;
  `persist-credentials: false`.
- **Release**: ships the GPU bundle (`rosnet-gpu-<tag>.cyr`); tag filter
  `X.Y.Z*`; write token only on the publishing job; actions pinned by SHA.
- `scripts/bench-history.sh` and `bench-history.csv` (before and after rows);
  benchmarks for every BLAS-1 op and conv1d; `.gitattributes` (`lib/** -text`).
- **`lib/agnosys.cyr` removed** — a scaffold-era file no cyrius snapshot since
  6.6.0 ships and nothing used, which `lib sync --full` never prunes. `lib/` is
  now exactly the 6.7.18 snapshot plus tyche; `cyrius.lock` records 114 files.

### Documentation

- **`docs/benchmarks.md` overstated conv throughput tenfold.** The conv2d shape
  is 3.61 M multiply-adds, not 36.1 M, so 1.1.0's conv ran at ~0.11 G MAC/s, not
  ~1.1 (the 1.1.0 entry below and `state.md` repeated it).
- `docs/api.md`: `linear_fwd` is an AXPY, not a tree-reduced dot;
  `gpu_matmul_fwd` and `gpu_ln_fwd` are bit-exact (gated on `g_gpu`), not
  tolerance-class; a GPU fallback may leave outputs partially written (the
  accumulated ones are written only after the last fallback point); new
  *Numerical reproducibility* section — x86_64 vs aarch64, where cyrius fuses
  the packed multiply-add.
- `SECURITY.md` described 0.2.0 and an unaudited library; rewritten for 1.x.
- `CLAUDE.md`: `var g[N]` at module scope is N i64 slots, not N bytes.
- ADR 0002; every public function documented (`cyrius doc --check` clean).

### Note — GPU consumers need cyrius ≥ 6.6.19

True since 1.1.3, found now. Newer distlib writes an `include "lib/mabda.cyr"`
prelude into the GPU bundle, which 6.6.2 – 6.6.18 cannot compile ("unexpected
enum" at 6.6.2, an undefined `MAP_SHARED` at 6.6.10 – 6.6.18). 6.6.19 and later
compile it with identical shader output. The CPU bundle is unaffected: the 1.2.2
CPU bundle compiles under 6.6.2 and gives the golden dump bit for bit. attn11
pins 6.6.2 and rosnet 1.1.2, so it must move to 6.6.19+ when it takes a GPU
bundle from 1.1.3 or later.

## [1.2.1] - 2026-11-09

A toolchain and dependency realign. No source change.

### Changed

- **Toolchain `6.7.9` → `6.7.18`** (through 6.7.10–6.7.17), the latest cyrius
  release. `src/` is unchanged, and so is every value rosnet computes: 1.2.0's
  7,462-word golden dump of raw f64 bit patterns is byte-identical at 6.7.18,
  with tyche 1.2.2 in place. The GPU profile, which cannot run without its
  device, gets a second dump. It runs each of the 24 SPIR-V shader builders in
  `src/gpu.cyr` (27 calls) into a host buffer and prints every emitted word,
  along with the Adam step's host-side `f64_pow` scalars and the GELU constant.
  That is 18,167 lines, byte-identical under 6.6.20, 6.7.9 and 6.7.18, so it
  also covers 1.1.3 and 1.2.0. **23 assertions** pass, the fuzz harness passes,
  and fmt and lint are clean, including 6.7.11's and 6.7.12's new lint
  warnings. Benchmarks did not move: on the same host, the medians of three
  interleaved runs per toolchain are within 1% on every op. The smoke binary
  shrinks from 175,248 to 174,752 B (31,888 to 31,392 B with DCE), all of it
  from the stdlib and the toolchain.
- **`[deps.tyche]` `1.2.0` → `1.2.2`**, the latest tyche, built against 6.7.18.
  Every state its API can reach gives the same bits as 1.2.0, and
  `rng_normal` is about 4% faster. Its one behaviour change is to a written
  `_rng_state` of 0, which no longer hangs `rng_normal`. rosnet never writes
  `_rng_state`.
- **Vendored `lib/` re-synced** (`cyrius lib sync --full`). It byte-matches the
  6.7.18 snapshot: the same 113 files as 6.7.9, 66 of them with new content.
  Nothing was added or removed.
- **`cyrius.lock` regenerated from scratch.** At 6.7.18, `cyrius deps` keeps
  the old tyche pin on purpose and says so ("keeps tyche@1.2.0's pin …,
  fail-closed; delete cyrius.lock to re-pin"). Deleting the lock and
  re-resolving leaves one tyche line (`cdb74e2`) and the same 115 file hashes,
  with trailer `cyrius 6.7.18`. `cyrius deps --verify` and
  `cyrius deps --locked` are clean.
- `dist/rosnet.cyr` and `dist/rosnet-gpu.cyr` change only in their version
  line, and both sidecars are unchanged. The GPU bundle still relies on the
  consumer's `puts` (see 1.1.3).

### Note — the pin, `lib/` and `cyrius.lock` land together

With a fully vendored `lib/`, cyrius refuses a moved pin until `lib/` follows
it. Moving 1.2.0's pin to 6.7.18 alone makes both `cyrius deps` and
`cyrius build` exit 1 with "cyrius.lock not re-locked: [package].cyrius moved
from 6.7.9 to 6.7.18, and lib/ holds 47 files … that the 6.7.18 snapshot ships
with other bytes", until `cyrius lib sync --full` runs. So a commit that moves
the pin in `cyrius.cyml` without `lib/` and `cyrius.lock` fails CI at its
`cyrius deps` step.

## [1.2.0] - 2026-11-09

Moves rosnet onto the cyrius 6.7 series. No library or API change.

### Changed

- **Toolchain `6.6.20` → `6.7.9`** (through 6.7.0–6.7.8). `src/` is unchanged,
  and so is every value rosnet computes. A dump of 7,462 raw f64 bit
  patterns is byte-identical between 1.1.3 @ 6.6.20 and 1.2.0 @ 6.7.9. It runs
  every CPU op at a fixed seed: BLAS-1 at length 37 (not a SIMD multiple);
  `linear_fwd` / `linear_bwd` at four shapes up to 16×96×33; `conv2d_fwd` /
  `conv2d_bwd` at three, strided, padded and non-square; and `conv1d_fwd` /
  `conv1d_bwd` at two. **23 assertions** pass, the fuzz harness passes, and fmt
  and lint are clean. Benchmarks did not slow down: on the same host, medians
  of three interleaved runs per toolchain are 0–5% faster at 6.7.9, inside the
  3–4% spread between runs of one toolchain. The smoke binary stays at
  175,248 B.

  Upstream changes checked against rosnet's source (`src/`, `tests/`):
  - **6.7.0–6.7.8** reserve `trait`, `const`, `true`, `false`, `do` and `dyn`,
    and **6.7.3** makes `!` a real operator. Before that a lone `!x` compiled as
    `x`. rosnet uses none of them outside comments and strings.
  - **6.7.9** makes unary `-` and `~` on a constant bind tighter, and
    **6.7.8–6.7.9** change how typed `*T` pointers step and dereference.
    rosnet has no unary `-` or `~` on a name, and no typed pointers: its
    tensors are untyped addresses read with `load64` / `store64`.
  - **6.7.6** rounds every write into an `f32`. rosnet has no `f32`.
  - **6.7.6** turns mabda's `F64_*` colour values into `const`s. The GELU
    constant in `src/gpu.cyr` reads `F64_PI`, which comes from `math.cyr` and
    was already initialised to π.
- **Tests run as `cyrius test tests`.** cyrius 6.7.6 deprecated `cyrius tests`,
  which now warns, and 6.7.12 removes it. CI's Tests step and the README now use
  `cyrius test tests`: same suite, same result. Bare `cyrius test`, which
  CLAUDE.md, `CONTRIBUTING.md` and the getting-started guide use, still runs it.
- **`[deps.tyche]` `1.1.1` → `1.2.0`**, the tyche release built against 6.7.9.
  tyche's source and stream did not change.
- **Vendored `lib/` re-synced** (`cyrius lib sync --full`). It byte-matches the
  6.7.9 snapshot: the same 113 files as 6.6.20, 67 of them with new content.
  Nothing was added or removed.
- **`cyrius.lock` regenerated from scratch**, as in 1.1.3: `cyrius deps` added
  the tyche 1.2.0 line (`2c393fe`) but kept the 1.1.1 line. The same 115 file
  hashes, trailer `cyrius 6.7.9`. `cyrius deps --verify` and 6.7.6's
  `cyrius deps --locked` are both clean.
- `dist/rosnet.cyr` and `dist/rosnet-gpu.cyr` change only in their version
  line, and both sidecars are unchanged. The GPU bundle still relies on the
  consumer's `puts` (see 1.1.3).

## [1.1.3] - 2026-11-09

A toolchain and dependency realign. No source change.

### Changed

- **Toolchain `6.6.2` → `6.6.20`** (through 6.6.3–6.6.19). **23 assertions**
  pass unchanged. fmt and lint are clean, and the smoke binary runs. The smoke
  binary grows from 109,856 to 175,248 B, all of it stdlib code that DCE would
  drop. Benchmarks did not move: two interleaved runs per toolchain on the same
  host put all six ops within the run-to-run spread of ±5%.
- **`[deps.tyche]` `1.0.1` → `1.1.1`**, the tyche release built against 6.6.20.
  tyche's frozen 1.x surface is unchanged. Its 1.1.0 changed `rng_normal` to use
  a portable logarithm, which moves 0.011% of x86_64 draws by 1–3 ulp and gives
  aarch64 the same stream as x86_64. `t_randn` inherits that. The suite uses
  `t_randn` only to fill grad-check inputs, never compares drawn values, and
  passes unchanged.
- **Vendored `lib/` re-synced** (`cyrius lib sync --full`). It byte-matches the
  6.6.20 snapshot, 113 files. Five are new: `alloc_cx`, `boxed`, `hashseed`,
  `poison` and `tls_hostid`. This clears the warning that `./lib/` shadowed 12
  pinned bundled libs, which 1.1.2 carried because it bumped the pin without a
  full re-sync.
- **`cyrius.lock` regenerated from scratch.** `cyrius deps` added the tyche
  1.1.1 line (`68bb5db`) but kept the 1.0.1 line (`73197d4`) above it.
  Regenerating leaves one tyche line, with the same 115 file hashes. The 6.6.20
  format also records the toolchain (`cyrius 6.6.20`). `cyrius deps --verify`
  is clean.
- **The sidecars list only what the bundles need.** From 6.6.18,
  `cyrius distlib` writes only the stdlib leaves a bundle provably needs
  (compile-verified) and also emits them as `include` lines at the top of the
  bundle.
  - `dist/rosnet.deps`: 10 → 4 leaves (`alloc`, `string`, `fmt`, `syscalls`).
  - `dist/rosnet-gpu.deps`: 14 → 4 leaves (`mabda`, `math`, `result`, `ganita`).

  **Consumers:** a build that used `io`, `vec`, `str`, `assert`, `bench` or
  `math` only because rosnet's sidecar brought it in must now declare that leaf
  in its own `[deps] stdlib`. cyrius 6.6.18 and later name the leaf to add when
  such a build fails. All six consumers already declare all ten leaves (checked
  in their manifests on 2026-11-09), so none is affected.
- **`cyrius distlib gpu` now succeeds.** The 1.1.2 note below said it errored
  on mabda and math enum constants the bundle leaves undefined. The gpu sidecar
  now pulls in `mabda`, `math` and `ganita`, so those constants resolve.
  `cyrius distlib --all --check` passes for both bundles.

### Note — the GPU bundle relies on the consumer's `puts`

6.6.20's compile-verify reports one name the GPU bundle uses that no stdlib
leaf defines: `puts`, which `gpu_probe` calls four times. No cyrius release
from 6.6.0 on defines it. attn11, the GPU consumer, defines its own (`fn puts`
in its `src/tensor.cyr`), so nothing broke. This was already true at 1.1.2,
whose toolchain did not check undefined functions, so this release changes
nothing for consumers. A GPU consumer that does not define `puts` cannot link
`gpu_probe`. It is not fixed here, because this release changes no source.

## [1.1.2] - 2026-09-11

### Fixed — the GPU bundle shipped a retired accessor

⛔ **Invisible to a build, found only by cyrius 6.6.2's new distlib check.**
`src/gpu.cyr` lives in the `[lib.gpu]` profile — not in `[lib]`, not in the
default entry — so `cyrius build` never compiles it and reported zero errors.
It called `payload(...)` twice (`_gpu_init` and `gpu_probe`), an accessor cyrius
**deleted at 6.6.0**. `dist/rosnet-gpu.cyr` shipped both, so any consumer
importing that bundle alongside `[deps.mabda]` could not have linked at 6.6.x.

6.6.2 surfaced it by refusing to emit the bundle at all:

```
error: distlib: the bundle calls a RETIRED stdlib name: dist/rosnet-gpu.cyr
      symbol: payload
```

Both sites now bind the `(tag, payload)` pair. The regenerated bundle scans zero
retired names.

### Changed

- **Toolchain `6.5.27` → `6.6.2`** — the largest pin jump in this sweep. The
  default profile (`tensor` / `linear` / `conv`) needed no change; **23
  assertions** pass unchanged.
- `dist/rosnet-gpu.deps` now lists its 14 stdlib leaves. It was empty apart from
  its two comment lines, so a consumer got no leaf requirements at all. All 14
  resolve against the 6.6.2 stdlib — including `mabda`, which is a *folded*
  stdlib module at 6.6.x, not a git dep, so listing it is correct.

### Note — `distlib gpu` still errors, by design

The gpu profile deliberately leaves mabda's device/dispatch symbols for the
consumer to supply (see the `[lib.gpu]` comment in `cyrius.cyml`). Undefined
*functions* are downgraded on the grounds that a bundle omits the stdlib — but
undefined *enum constants* are not, so `MABDA_MAX_DISPATCH_DIM`, `_NATIVE_PERM_DATA`
and `F64_PI` make 6.6.2's stricter check report the bundle as non-compiling. The
bundle is still emitted and is correct for its documented use. CI gates only
`cyrius distlib` (the default profile), so nothing here is red — but anyone
running `cyrius distlib gpu` by hand will see the error.


## [1.1.1] - 2026-08-17

### Changed

- **Cyrius pin `6.2.11` -> `6.5.27`** (2026-08-17, ecosystem-wide ML/AI-arc realign ahead of
  the arc reopening). `cyrius lib sync --full` re-vendored the whole version-matched stdlib
  snapshot, clearing the toolchain-drift and `./lib/ shadows version-pinned` warnings.
  Retires the deliberate deep-lag recorded in `docs/development/state.md` — the consumer-driven touch arrived. Suite **23/23**, identical to the pre-bump baseline; the 6.3.13 array-locals-to-stack change was safe here as predicted (the SIMD accumulators are 32-byte locals). Cleared a 12-module shadow warning.
- **`[deps.tyche]` `0.1.1` -> `1.0.1`** — the first step of the substrate-freeze propagation.
  tyche cut **1.0.0** on 2026-07-05 and froze its 1.x surface (`rng_seed` / `rng_u64` /
  `rng_uniform` / `rng_normal` + the documented `_rng_state` checkpoint exception), and rosnet
  — one of the four consumers whose unchanged use *was* the readiness evidence for that freeze
  — had never moved off the pre-1.0 tag. No behavior change: `t_randn`'s only call is
  `rng_normal`, which is in the frozen set. Verified the bump actually took rather than merely
  built — vendored `lib/tyche.cyr` header moved to `# Version: 1.0.1`, `cyrius.lock` resolved
  tag `1.0.1` at commit `73197d4`, and its recorded content hash matches the file on disk.
  Suite **23/23**, unchanged.

## [1.1.0] - 2026-07-05

**conv2d/conv1d with hand-derived gradients — the 1.1 additive.** The last new
LEARNED primitive gating the modality axis (sight + hearing) and the Type-4
diffusion reference (per agnosticos `planning/multimodal-substrate.md` — every
other frontend piece already landed: chitra decode, hisab FFT, rupantara
blocks).

### Added
- **`src/conv.cyr`** (joins the CPU `[lib]` bundle, 245→390 dist lines):
  `conv2d_fwd(x, Wf, b, y, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW)` +
  `conv2d_bwd(...)` — NCHW row-major, **per-axis stride/padding** (so audio
  stems stride along time only), zero-padding, optional bias; backward follows
  the linear_bwd convention (dx written — pass 0 to skip, e.g. a data-input
  first layer; dW/db accumulated) with all three gradients hand-derived
  (dW = x⊛dy gather; dx = the transpose-conv scatter; db = Σdy).
  `conv1d_fwd/bwd` = exact H=1 wrappers (no phantom padding on the dummy
  axis — the per-axis design exists precisely for this), bit-identical to the
  2D path by construction.
- **FD gates** (suite 7→**23**): known answers (1×1 kernel affine identity,
  sum-kernel, padding-corner arithmetic) + central-finite-difference checks of
  dx/dW/db (rel < 1e-5, the linear_bwd gate) across three configurations —
  multi-batch/channel 3×3, **strided+padded non-square** (2×3 kernel, S 2×2,
  P 1×1), and a 1×K row case — plus conv1d↔conv2d bit-parity.
- **Benchmarks** (`docs/benchmarks.md`): 8×8×28×28→8 (3×3, pad 1) fwd
  **32.4 ms** / bwd **43.3 ms** (~1.1 G MAC/s — the direct loops hold the
  substrate's linear-kernel rate; im2col/SIMD is the named later
  optimization, fed by the in-progress cyrius SIMD arc when it releases).

### Notes
- Direct-convolution reference (deterministic accumulation order, no scratch)
  — correctness-first per the substrate's honest framing.
- API-freeze status: **additive 1.x surface**, now frozen with this cut
  (`docs/api.md`); the 1.0 surface is untouched.

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
