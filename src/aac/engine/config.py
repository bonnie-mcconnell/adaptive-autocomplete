"""
Engine configuration serialisation with predictor registry.

Problem
-------
``AutocompleteEngine`` is configured in Python code: predictor classes,
weights, ranker classes, vocabulary, history path.  Deploying the same
engine to multiple servers means repeating the constructor call or writing
a shared factory function.  There is no way to inspect a running engine's
configuration, diff two configurations, or store a configuration alongside
the vocabulary.

Solution
--------
``EngineConfig`` is a JSON-serialisable representation of an engine's full
configuration.  ``engine.to_config()`` serialises a running engine.
``EngineConfig.from_json(...).build()`` reconstructs it.

For preset engines, ``build()`` delegates to ``create_engine()`` - fast and
exact.  For custom engines, ``build()`` uses the ``PredictorRegistry`` to
resolve predictor classes by name.  Third-party predictors can be registered
via ``PredictorRegistry.register()``.

What is serialised
------------------
- Preset name (if the engine was built via ``create_engine()``)
- Predictor names and weights
- Ranker names and parameters (half-life, weight)
- The engine's schema version

What is NOT serialised
----------------------
- History (use ``JsonHistoryStore.save()`` for that - it is a separate concern)
- Vocabulary (too large; reference the vocabulary path instead)
- Internal index state (rebuilt at ``from_config()`` time)

Usage
-----
::

    from aac.presets import create_engine
    from aac.engine.config import EngineConfig

    engine = create_engine("production")
    config = engine.to_config()

    # Reconstruct on another server
    engine2 = EngineConfig.from_json(config.to_json()).build()
    assert engine2.suggest("prog") == engine.suggest("prog")

    # Custom engine reconstruction via registry
    from aac.engine.config import PredictorRegistry

    class MyPredictor:
        name = "my_predictor"
        def predict(self, ctx): ...

    PredictorRegistry.register("my_predictor", lambda vocab, params: MyPredictor())
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aac.domain.history import History
    from aac.domain.types import Predictor, WeightedPredictor
    from aac.engine.engine import AutocompleteEngine
    from aac.ranking.base import Ranker

_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Predictor registry
# ---------------------------------------------------------------------------

# Factory signature: (vocabulary, params) -> Predictor instance.
# params is a plain dict so factories can accept arbitrary keyword config
# without changing the registry interface. History-aware factories check
# params.get("_history") - a private key injected by build_predictor().
_PredictorFactory = Callable[[Mapping[str, int] | None, dict[str, Any]], "Predictor"]


class PredictorRegistry:
    """
    Registry mapping predictor names to factory callables.

    Allows ``EngineConfig.build()`` to reconstruct custom engines from
    config without hardcoding all predictor classes.

    Built-in predictors (frequency, history, symspell, trigram, bktree,
    trie, static_prefix) are registered automatically at import time.
    Third-party or user-defined predictors must be registered before
    calling ``build()``.

    Example::

        from aac.engine.config import PredictorRegistry

        class MyDomainPredictor:
            name = "domain"
            def predict(self, ctx): ...

        PredictorRegistry.register(
            "domain",
            lambda vocab, params: MyDomainPredictor()
        )
    """

    _registry: dict[str, _PredictorFactory] = {}

    @classmethod
    def register(cls, name: str, factory: _PredictorFactory) -> None:
        """
        Register a factory for a predictor name.

        Parameters:
            name:    The predictor's ``name`` attribute (used as the config key).
            factory: Callable ``(vocabulary, params) -> Predictor``.
                     ``vocabulary`` is the dict passed to ``build()``, or None.
                     ``params`` is the ``params`` dict from ``PredictorConfig``.
        """
        cls._registry[name] = factory

    @classmethod
    def build_predictor(
        cls,
        name: str,
        vocabulary: Mapping[str, int] | None,
        params: dict[str, Any],
        history: History | None = None,
    ) -> Predictor:
        """
        Build a predictor instance by name.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in cls._registry:
            registered = sorted(cls._registry.keys())
            raise KeyError(
                f"No predictor registered for name {name!r}. "
                f"Registered predictors: {registered}. "
                f"Use PredictorRegistry.register() to add custom predictors."
            )
        factory = cls._registry[name]
        # History-aware predictors receive history via params using a
        # private key.  The factory signature is (vocabulary, params);
        # factories that need history should check params.get('_history').
        if history is not None:
            params = {**params, "_history": history}
        return factory(vocabulary, params)

    @classmethod
    def registered_names(cls) -> list[str]:
        """Return sorted list of all registered predictor names."""
        return sorted(cls._registry.keys())


def _register_builtins() -> None:
    """Register all built-in predictors.  Called once at module import."""
    from aac.domain.history import History
    from aac.predictors.edit_distance import EditDistancePredictor
    from aac.predictors.frequency import FrequencyPredictor
    from aac.predictors.history import HistoryPredictor
    from aac.predictors.static_prefix import StaticPrefixPredictor
    from aac.predictors.symspell import SymSpellPredictor
    from aac.predictors.trie import TriePrefixPredictor
    from aac.predictors.trigram import TrigramPredictor

    def _freq(vocab: Mapping[str, int] | None, params: dict[str, Any]) -> Predictor:
        from aac.data import load_english_frequencies
        frequencies = vocab or load_english_frequencies()
        return FrequencyPredictor(
            frequencies,
            max_results=params.get("max_results", 100),
        )

    def _history_pred(vocab: Mapping[str, int] | None, params: dict[str, Any]) -> Predictor:
        raw_history = params.get("_history")
        history = raw_history if isinstance(raw_history, History) else History()
        return HistoryPredictor(history)

    def _symspell(vocab: Mapping[str, int] | None, params: dict[str, Any]) -> Predictor:
        from aac.data import load_english_frequencies
        frequencies = vocab or load_english_frequencies()
        return SymSpellPredictor(
            list(frequencies.keys()),
            max_distance=params.get("max_distance", 2),
            frequencies=frequencies,
        )

    def _trigram(vocab: Mapping[str, int] | None, params: dict[str, Any]) -> Predictor:
        from aac.data import load_english_frequencies
        frequencies = vocab or load_english_frequencies()
        return TrigramPredictor(
            list(frequencies.keys()),
            max_distance=params.get("max_distance", 2),
            frequencies=frequencies,
        )

    def _bktree(vocab: Mapping[str, int] | None, params: dict[str, Any]) -> Predictor:
        from aac.data import load_english_frequencies
        frequencies = vocab or load_english_frequencies()
        return EditDistancePredictor(
            list(frequencies.keys()),
            max_distance=params.get("max_distance", 2),
            frequencies=frequencies,
        )

    def _trie(vocab: Mapping[str, int] | None, params: dict[str, Any]) -> Predictor:
        from aac.data import load_english_frequencies
        frequencies = vocab or load_english_frequencies()
        return TriePrefixPredictor(list(frequencies.keys()))

    def _static(vocab: Mapping[str, int] | None, params: dict[str, Any]) -> Predictor:
        from aac.data import load_english_frequencies
        frequencies = vocab or load_english_frequencies()
        return StaticPrefixPredictor(list(frequencies.keys()))

    def _adaptive_symspell(vocab: Mapping[str, int] | None, params: dict[str, Any]) -> Predictor:
        from aac.data import load_english_frequencies
        from aac.predictors.adaptive_symspell import AdaptiveSymSpellPredictor
        frequencies = vocab or load_english_frequencies()
        return AdaptiveSymSpellPredictor(
            list(frequencies.keys()),
            max_distance=params.get("max_distance", 2),
            frequencies=frequencies,
        )

    PredictorRegistry.register("frequency", _freq)
    PredictorRegistry.register("history", _history_pred)
    PredictorRegistry.register("symspell", _symspell)
    PredictorRegistry.register("adaptive_symspell", _adaptive_symspell)
    PredictorRegistry.register("trigram", _trigram)
    PredictorRegistry.register("bktree", _bktree)
    PredictorRegistry.register("trie", _trie)
    PredictorRegistry.register("static_prefix", _static)


# Register builtins immediately so any import of this module gives
# a usable registry without requiring an explicit initialisation call.
_register_builtins()


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PredictorConfig:
    """Serialisable predictor configuration."""
    name: str
    weight: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankerConfig:
    """Serialisable ranker configuration."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineConfig:
    """
    JSON-serialisable engine configuration.

    Captures the full structural configuration of an ``AutocompleteEngine``
    so it can be reconstructed on another process or server without
    repeating Python constructor calls.

    Attributes:
        preset:      Name of the preset used to build the engine, or None
                     for custom engines.
        predictors:  Predictor names and weights in order.
        rankers:     Ranker names and parameters in order.
        version:     Schema version for forward compatibility.
        metadata:    Arbitrary caller-supplied key-value pairs (e.g.
                     ``{"vocabulary_path": "...", "deployed_at": "..."}``).

    Example::

        config = EngineConfig(
            preset="production",
            predictors=[
                PredictorConfig("frequency", weight=1.0),
                PredictorConfig("history",   weight=1.2),
                PredictorConfig("symspell",  weight=0.35),
                PredictorConfig("trigram",   weight=0.4),
            ],
            rankers=[
                RankerConfig("score"),
                RankerConfig("decay", params={"half_life_seconds": 3600, "weight": 1.5}),
            ],
            metadata={"vocabulary_path": "~/.aac_vocab.json"},
        )
    """

    preset: str | None
    predictors: list[PredictorConfig]
    rankers: list[RankerConfig]
    version: int = _SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "version": self.version,
            "preset": self.preset,
            "predictors": [asdict(p) for p in self.predictors],
            "rankers": [asdict(r) for r in self.rankers],
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineConfig:
        """Deserialise from a dict (e.g. from ``json.loads()``)."""
        version = data.get("version", 1)
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported EngineConfig schema version {version!r}. "
                f"Expected {_SCHEMA_VERSION}. "
                f"Migrate the config or downgrade the library."
            )
        return cls(
            preset=data.get("preset"),
            predictors=[
                PredictorConfig(
                    name=p["name"],
                    weight=p["weight"],
                    params=p.get("params", {}),
                )
                for p in data.get("predictors", [])
            ],
            rankers=[
                RankerConfig(
                    name=r["name"],
                    params=r.get("params", {}),
                )
                for r in data.get("rankers", [])
            ],
            version=version,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> EngineConfig:
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(text))

    def build(
        self,
        vocabulary: dict[str, int] | None = None,
        history: History | None = None,
    ) -> AutocompleteEngine:
        """
        Reconstruct an ``AutocompleteEngine`` from this config.

        For **preset engines** (``preset`` is set), delegates to
        ``create_engine()`` - this is the fast path and produces identical
        behaviour to the original engine.

        For **custom engines** (``preset`` is None), uses the
        ``PredictorRegistry`` to resolve predictor classes by name and
        rebuilds the engine from its structural config.  All built-in
        predictors are registered automatically; custom predictors must be
        registered via ``PredictorRegistry.register()`` before calling
        ``build()``.

        Parameters:
            vocabulary: Custom vocabulary mapping word → frequency count.
                        If None and ``metadata`` contains a ``"vocabulary_path"``
                        key, a warning is emitted.  If None with no metadata
                        hint, the bundled 48k English corpus is used.
            history:    Pre-loaded ``History`` instance.  If None, starts
                        with an empty in-memory history.

        Returns:
            A fully initialised ``AutocompleteEngine``.

        Example::

            config = engine.to_config(
                metadata={"vocabulary_path": str(vocab_path)},
            )
            # ... round-trip via JSON ...
            import json
            vocab = json.load(open(config.metadata["vocabulary_path"]))
            engine2 = config.build(vocabulary=vocab)
        """
        import warnings

        from aac.presets import create_engine

        if vocabulary is None and "vocabulary_path" in self.metadata:
            warnings.warn(
                f"EngineConfig.build() was called without a vocabulary, but "
                f"metadata contains 'vocabulary_path': "
                f"{self.metadata['vocabulary_path']!r}. "
                f"The engine will use the bundled English corpus instead of "
                f"the original vocabulary. Pass vocabulary= explicitly to "
                f"restore exact behaviour.",
                UserWarning,
                stacklevel=2,
            )

        if self.preset is not None:
            return create_engine(self.preset, vocabulary=vocabulary, history=history)

        # Custom engine path: use the registry to reconstruct predictors.
        from aac.domain.history import History
        from aac.domain.types import WeightedPredictor
        from aac.engine.engine import AutocompleteEngine
        from aac.ranking.decay import DecayFunction, DecayRanker
        from aac.ranking.learning import LearningRanker
        from aac.ranking.score import ScoreRanker

        resolved_history: History = history if history is not None else History()

        predictors: list[WeightedPredictor] = []
        for pc in self.predictors:
            predictor = PredictorRegistry.build_predictor(
                pc.name,
                vocabulary,
                pc.params,
                history=resolved_history,
            )
            predictors.append(WeightedPredictor(predictor=predictor, weight=pc.weight))

        rankers: list[Ranker] = []
        for rc in self.rankers:
            if rc.name in ("score", "scoreranker"):
                rankers.append(ScoreRanker())
            elif rc.name in ("decay", "decayranker"):
                half_life = rc.params.get("half_life_seconds", 3600.0)
                weight = rc.params.get("weight", 1.0)
                rankers.append(
                    DecayRanker(
                        resolved_history,
                        DecayFunction(half_life_seconds=half_life),
                        weight=weight,
                    )
                )
            elif rc.name in ("learning", "learningranker"):
                boost = rc.params.get("boost", 1.0)
                dominance_ratio = rc.params.get("dominance_ratio", 1.0)
                rankers.append(
                    LearningRanker(
                        resolved_history,
                        boost=boost,
                        dominance_ratio=dominance_ratio,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown ranker {rc.name!r}. "
                    f"Known rankers: score, decay, learning."
                )

        return AutocompleteEngine(
            predictors=predictors,
            ranker=rankers or None,
            history=resolved_history,
        )

    def diff(self, other: EngineConfig) -> list[str]:
        """
        Return a list of human-readable differences between two configs.

        Returns an empty list if the configs are equivalent.  Useful for
        auditing whether two deployed engines have the same configuration.

        Example::

            changes = config_a.diff(config_b)
            if changes:
                print("Config changed:")
                for line in changes:
                    print(" ", line)
        """
        diffs: list[str] = []

        if self.preset != other.preset:
            diffs.append(f"preset: {self.preset!r} → {other.preset!r}")

        self_preds = {p.name: p for p in self.predictors}
        other_preds = {p.name: p for p in other.predictors}

        for name in set(self_preds) | set(other_preds):
            a = self_preds.get(name)
            b = other_preds.get(name)
            if a is None:
                # b cannot be None here: the loop iterates set(self_preds) | set(other_preds),
                # so every name is in at least one dict.  If a is None, b must be present.
                # Programming error (violated loop invariant), not user input,
                # so a RuntimeError is appropriate - but we avoid bare assert which is
                # stripped by -O and provides no message.
                if b is None:  # pragma: no cover
                    raise RuntimeError(
                        f"diff() invariant violated: predictor {name!r} missing from both configs"
                    )
                diffs.append(f"predictor added: {name!r} (weight={b.weight})")
            elif b is None:
                diffs.append(f"predictor removed: {name!r} (was weight={a.weight})")
            elif abs(a.weight - b.weight) > 1e-9:
                diffs.append(f"predictor {name!r} weight: {a.weight} → {b.weight}")

        self_rankers: list[str] = [r.name for r in self.rankers]
        other_rankers: list[str] = [r.name for r in other.rankers]
        if self_rankers != other_rankers:
            diffs.append(f"rankers: {self_rankers} → {other_rankers}")
        else:
            for a_r, b_r in zip(self.rankers, other.rankers, strict=False):
                if a_r.params != b_r.params:
                    diffs.append(
                        f"ranker {a_r.name!r} params: {a_r.params} → {b_r.params}"
                    )

        return diffs

    def __repr__(self) -> str:
        pred_summary = ", ".join(f"{p.name}×{p.weight}" for p in self.predictors)
        return (
            f"EngineConfig(preset={self.preset!r}, "
            f"predictors=[{pred_summary}], "
            f"rankers={[r.name for r in self.rankers]!r})"
        )
