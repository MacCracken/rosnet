# rosnet — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.1.1** — scaffolded 2026-06-11 via `cyrius init`.

## Toolchain

- **Cyrius pin**: `6.2.11` (in `cyrius.cyml [package].cyrius`)

## Source

Initial scaffold only.

## Tests

- `tests/rosnet.tcyr` — primary suite (smoke + math; passes on `cyrius test`)
- `tests/rosnet.bcyr` — benchmark stub (no-op)
- `tests/rosnet.fcyr` — fuzz stub

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert

## Consumers

_None yet._

## Next

See [`roadmap.md`](roadmap.md).
