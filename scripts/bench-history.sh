#!/usr/bin/env bash
# bench-history.sh — run rosnet's benchmarks and append the results to
# bench-history.csv, one row per benchmark, stamped with the date and VERSION.
#
# This is the baseline tool of the first-party P(-1) process: run it before a
# hardening or optimization pass and again after, and compare the two stamps.
# Adapted from tyche's script of the same name.
#
# Each benchmark is the median of RUNS runs (default 3). On a laptop-class host
# the same binary moves by a few percent between runs, so a single run can hide
# or invent a difference. Compare rows only when they were recorded in the same
# session on the same host; docs/benchmarks.md says which rows were.
#
# It runs the benchmark exactly as CI does:
#   CYRIUS_DCE=1 cyrius bench tests/rosnet.bcyr
# and parses lib/bench.cyr's `bench_report` line, whose label may contain
# spaces, parentheses and commas:
#   "  conv2d_fwd 8x8x28x28 -> 8 (3x3, pad 1): 32.056ms avg (min=… max=…) [10 iters]"
# The label is written to the CSV in double quotes.
set -euo pipefail

cd "$(dirname "$0")/.."

CSV="bench-history.csv"
BENCH="tests/rosnet.bcyr"
RUNS="${RUNS:-3}"
case "$RUNS" in
    ''|*[!0-9]*) echo "RUNS must be a positive integer, got '$RUNS'" >&2; exit 2 ;;
esac
[ "$RUNS" -ge 1 ] || { echo "RUNS must be at least 1" >&2; exit 2; }

DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
VERSION="$(tr -d '[:space:]' < VERSION)"

if [ ! -f "$CSV" ]; then
    echo "date,version,benchmark,time_ns" > "$CSV"
fi

TMPDIR_B="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_B"' EXIT

# One "<label>\t<ns>" line per benchmark per run. The pattern is anchored on the
# ": <value><unit> avg" shape, so compiler notes on the same stream are ignored.
i=1
while [ "$i" -le "$RUNS" ]; do
    echo "Run $i of $RUNS: CYRIUS_DCE=1 cyrius bench $BENCH"
    CYRIUS_DCE=1 cyrius bench "$BENCH" > "$TMPDIR_B/out" 2>&1 \
        || { cat "$TMPDIR_B/out" >&2; echo "cyrius bench failed" >&2; exit 1; }
    grep -E '^[[:space:]]+[A-Za-z_][^:]*: [0-9.]+(ns|us|ms|s) avg' "$TMPDIR_B/out" \
        | sed -E 's/^[[:space:]]*([A-Za-z_][^:]*):[[:space:]]*([0-9.]+)(ns|us|ms|s) avg.*/\1\t\2\t\3/' \
        | awk -F '\t' '{ s = ($3 == "ns") ? 1 : ($3 == "us") ? 1e3 : ($3 == "ms") ? 1e6 : 1e9;
                         printf "%s\t%.0f\n", $1, $2 * s }' >> "$TMPDIR_B/rows"
    i=$((i + 1))
done

[ -s "$TMPDIR_B/rows" ] || { echo "no benchmark results parsed" >&2; exit 1; }

# Median per benchmark, in first-seen order. With an even RUNS it takes the
# lower middle value, so every recorded number is one that was measured.
awk -F '\t' -v date="$DATE" -v ver="$VERSION" '
    { if (!($1 in n)) order[++k] = $1; v[$1, ++n[$1]] = $2 }
    END {
        for (j = 1; j <= k; j++) {
            name = order[j]; m = n[name]
            for (a = 1; a <= m; a++) x[a] = v[name, a]
            for (a = 2; a <= m; a++) { t = x[a]; b = a - 1
                while (b >= 1 && x[b] > t) { x[b + 1] = x[b]; b-- } x[b + 1] = t }
            gsub(/"/, "\"\"", name)
            printf "%s,%s,\"%s\",%d\n", date, ver, name, x[int((m + 1) / 2)]
        }
    }' "$TMPDIR_B/rows" | tee -a "$CSV"

echo "Appended $(grep -c "^${DATE}," "$CSV") rows to $CSV (median of $RUNS runs)"
