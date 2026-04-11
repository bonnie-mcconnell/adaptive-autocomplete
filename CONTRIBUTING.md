# Contributing

## Setup

```bash
git clone https://github.com/bonnie-mcconnell/adaptive-autocomplete
cd adaptive-autocomplete
make install
```

## Development workflow

```bash
make test       # run full test suite with coverage
make lint       # ruff check
make typecheck  # mypy --strict
make check      # lint + typecheck + test (run before every commit)
```

## Adding a predictor

1. Create `src/aac/predictors/your_predictor.py` implementing the `Predictor` protocol from `aac.domain.types`
2. Export it from `src/aac/predictors/__init__.py`
3. Add a contract test class to `tests/contracts/test_predictor_contracts.py` subclassing `PredictorContractTestMixin`
4. Write unit tests in `tests/predictors/`

The `Predictor` protocol requires `name: str` and `predict(ctx: CompletionContext) -> list[ScoredSuggestion]`. See `FrequencyPredictor` for a well-documented example.

## Adding a ranker

1. Create `src/aac/ranking/your_ranker.py` subclassing `Ranker` from `aac.ranking.base`
2. Implement `rank()` and `explain()`
3. The engine enforces: rankers must not add or remove suggestions. Tests in `tests/ranking/` verify this contract.

If your ranker learns from history, also implement `LearnsFromHistory` from `aac.ranking.contracts`.

## Test standards

- Every new module needs tests
- Tests must cover edge cases, not just happy paths - empty inputs, boundary conditions, invariant violations
- Correctness tests for search/matching components should compare against a brute-force reference (see `test_bk_tree.py` and `test_trigram_predictor.py`)
- New rankers must have an explain invariant test: `final_score == base_score + history_boost`

## Code standards

- Full type annotations; `mypy --strict` must pass
- `ruff check` must pass
- No bare `except`; no silent failures
- `assert` is not used for runtime invariants (disabled under `-O`); use `RuntimeError` or `ValueError`
