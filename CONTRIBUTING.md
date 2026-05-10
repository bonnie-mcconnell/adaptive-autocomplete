# Contributing

## Setup

```bash
git clone https://github.com/bonnie-mcconnell/adaptive-autocomplete
cd adaptive-autocomplete
make dev-setup   # installs dependencies + pre-commit hooks (run once)

# Activate the virtualenv to use 'aac' directly:
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell

# Or prefix any command with 'poetry run':
poetry run aac suggest he
```

## Development workflow

```bash
make test-fast  # unit tests only - fast, no subprocess overhead (~30s)
make test       # full suite including integration tests (~90s)
make lint       # ruff check
make typecheck  # mypy --strict
make check      # lint + typecheck + full test suite (run before every commit)
make pre-commit # run all pre-commit hooks against all files
```

Integration tests are marked with `@pytest.mark.integration` and invoke the
CLI as a subprocess. They are slow but exercise the full stack end-to-end.
Run them before opening a PR. CI runs unit tests on every push and the full
suite on every pull request.

## Adding a predictor

1. Create `src/aac/predictors/your_predictor.py` implementing the `Predictor` protocol from `aac.domain.types`
2. Export it from `src/aac/predictors/__init__.py`
3. Register it in `_register_builtins()` in `src/aac/engine/config.py` so it's accessible by name via `EngineConfig`
4. Add a contract test class to `tests/contracts/test_predictor_contracts.py` subclassing `PredictorContractTestMixin`
5. Write unit tests in `tests/predictors/`

The `Predictor` protocol requires `name: str` and `predict(ctx: CompletionContext) -> list[ScoredSuggestion]`. See `FrequencyPredictor` for a well-documented example.

If your predictor uses distance scoring (edit distance, n-gram similarity, etc.), use `distance_score()` and `edit_confidence()` from `aac.predictors._scoring` - **do not** inline the formula. All three existing distance predictors use this shared module so their scores are directly comparable in a weighted stack.

## Adding a ranker

Rankers modify the ordering of suggestions after the weighted score aggregation step. They have access to `History` and can apply time-decay, learning, or any other re-ranking signal.

1. Create `src/aac/ranking/your_ranker.py` subclassing `Ranker` from `aac.ranking.base`
2. Implement `rank(prefix, suggestions) -> list[ScoredSuggestion]` and `explain(prefix, suggestions) -> list[RankingExplanation]`
3. If your ranker binds to `History` at construction time:
   - Register it in `WeightOptimiser._rebuild_rankers_for_history()` in `src/aac/evaluation/optimiser.py` so weight optimisation can build fresh instances per evaluation
   - Register it in `EngineConfig.build()` in `src/aac/engine/config.py` so JSON-serialised engines can reconstruct it
4. Add a contract test to `tests/contracts/` if you add a `RankerContractTestMixin` (currently missing - good first contribution)
5. Write unit tests in `tests/ranking/`

Key invariant: `explain()` must return suggestions in the same order as `rank()`. The engine enforces this at runtime via `test_explain_ordering_agreement.py`. Your ranker must not reorder between `rank()` and `explain()`.

Key warning: if your ranker reads from `History`, implement `LearnsFromHistory` from `aac.ranking.contracts` by exposing a `history` attribute. The engine uses this to validate that all rankers and the engine share the same `History` instance at construction time.

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
