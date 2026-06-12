# rosnet

Sovereign **dense f64 tensor algebra** — the BLAS substrate for
[AGNOS](https://github.com/MacCracken/agnosticos), written in
[Cyrius](https://github.com/MacCracken/cyrius). Storage, BLAS-1 vector ops, and
the dense matmul **with its gradient** — the reusable numeric core under any
gradient-based workload, no BLAS / libc / autodiff.

A tensor is just a heap pointer to a flat, row-major run of f64 values (IEEE-754
bit patterns carried in `i64`); shapes live in the caller. Extracted from
[attn11](https://github.com/MacCracken/attn11)'s tensor layer at its v1 freeze.

## API

**Storage / element access** — `t_alloc(n)`, `t_zero`, `t_fill`, `t_copy`,
`tget(p,i)`, `tset(p,i,v)`, `ff(n)` (f64 from int literal).

**BLAS-1** — `t_axpy(dst,src,a,n)` (`dst += a·src`), `t_add_into`, `t_scale`,
`t_sum`.

**BLAS-3 (matmul + gradient)** — pure linear algebra, validated against central
finite differences:
- `linear_fwd(x, W, b, y, M, K, N)` — `y(M,N) = x(M,K) · W(K,N) [+ b(N)]`, 4-wide SIMD.
- `linear_bwd(x, W, dy, dx, dW, db, M, K, N)` — writes `dx`, accumulates `dW`/`db` (`dx = dy·Wᵀ`, `dW = xᵀ·dy`). `b`/`db == 0` means "no bias".

**Guards / init** — `f64_is_finite(x)` (bit-pattern NaN/Inf check),
`t_randn(p, n, std)` (normal-fill; see dependency note).

## Dependency: tyche

`t_randn` draws from [tyche](https://github.com/MacCracken/tyche)'s deterministic
statistical PRNG (`rng_normal`). It is the **only** rosnet symbol that needs
tyche — the pure tensor/matmul ops do not. In the `dist/rosnet.cyr` bundle
`rng_normal` is left unresolved, so a consumer that calls `t_randn` must also:

```toml
[deps.tyche]
git = "https://github.com/MacCracken/tyche"
tag = "0.1.0"
modules = ["dist/tyche.cyr"]
```

Seed the stream with tyche's `rng_seed(s)` before `t_randn` for reproducible
fills. (tyche is a *statistical* PRNG — never use it for crypto; that's sigil.)

## Use as a dependency

```toml
[deps.rosnet]
git = "https://github.com/MacCracken/rosnet"
tag = "0.1.0"
modules = ["dist/rosnet.cyr"]
```

Then `cyrius deps` and `include "lib/rosnet.cyr"`.

## Build

```sh
cyrius deps                                # resolve tyche + stdlib
cyrius build src/main.cyr build/rosnet     # compile the smoke demo
cyrius tests                               # grad-checks + BLAS-1 (tests/*.tcyr)
cyrius distlib                             # regenerate dist/rosnet.cyr
```

## License

GPL-3.0-only
