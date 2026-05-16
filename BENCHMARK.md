# Benchmarks

Latency numbers for `aac` on the bundled 48k-word English vocabulary.
All figures are wall-clock time measured on Ubuntu 24.04, Python 3.12,
single core, no warm-up discarded. Numbers are p50 / p95 / p99 across
10,000 calls per preset.

CI enforces hard p99 gates (see `tests/test_performance_regression.py`).
The table below is informational; numbers vary by machine.

## suggest() latency

| Preset | p50 | p95 | p99 | Gate |
|---|---|---|---|---|
| `stateless` | ~0.05 ms | ~0.07 ms | ~0.10 ms | p99 < 5 ms |
| `default` | ~0.10 ms | ~0.15 ms | ~0.20 ms | p99 < 5 ms |
| `recency` | ~0.12 ms | ~0.18 ms | ~0.25 ms | p99 < 5 ms |
| `robust` | ~1.8 ms | ~2.2 ms | ~2.8 ms | p99 < 30 ms |
| `production` | ~2.1 ms | ~2.6 ms | ~3.2 ms | p99 < 30 ms |

The CI gate is 15–20× the measured value on a typical GitHub Actions runner,
so it catches catastrophic regressions without being fragile to runner variance.

## Index build time

| Preset | Build time |
|---|---|
| `stateless`, `default`, `recency` | < 0.1 s |
| `robust` (SymSpell) | ~2 s |
| `production` (AdaptiveSymSpell + Trigram) | ~4–5 s |

Build time is paid once at startup. `WeightOptimiser` caches indexes across
evaluations so tuning 27 weight combinations on `production` costs ~0.05s
total, not ~135s.

## explain() overhead

`explain()` runs one pipeline pass, the same as `suggest()`. Overhead vs
`suggest()` is the delta-capture bookkeeping - typically < 5% additional
latency.

## Reproducing locally

```bash
make benchmark
# or
poetry run python -m aac.benchmarks.benchmark_engine
```

CI runs benchmarks on Python 3.12 / ubuntu-latest and uploads results as an
artifact (`benchmark-results-<sha>`). Check the Actions tab for trend data.

## Environment

Reproducible environment for controlled comparison:

```bash
# Pin CPU (Linux)
taskset -c 0 poetry run python -m aac.benchmarks.benchmark_engine

# Docker (fixed resource limits)
docker run --cpus=1 --memory=512m adaptive-autocomplete \
    python -m aac.benchmarks.benchmark_engine
```
