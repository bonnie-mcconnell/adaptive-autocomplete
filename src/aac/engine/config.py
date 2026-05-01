"""
Engine configuration serialisation.

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
``AutocompleteEngine.from_config(config)`` reconstructs it.

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
    print(config.to_json())

    # Reconstruct on another server
    engine2 = EngineConfig.from_json(config.to_json()).build()
    assert engine2.suggest("prog") == engine.suggest("prog")

    # Save alongside vocabulary
    import json
    with open("engine_config.json", "w") as f:
        f.write(config.to_json())

    with open("engine_config.json") as f:
        engine3 = EngineConfig.from_json(f.read()).build()
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

_SCHEMA_VERSION = 1


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
                     ``{"vocabulary_path": "...", "deployed_at": "..."}``)
                     stored alongside the config for operational use.

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
        history: Any = None,
    ) -> Any:
        """
        Reconstruct an ``AutocompleteEngine`` from this config.

        If ``preset`` is set, delegates to ``create_engine()`` - this is
        the fast path and produces identical behaviour to the original engine,
        provided the same vocabulary is supplied.

        If ``preset`` is None (custom engine), raises ``NotImplementedError``
        - custom engine reconstruction requires caller-supplied predictor
        and ranker instances that cannot be inferred from names alone.

        Parameters:
            vocabulary: Custom vocabulary mapping word → frequency count.
                        If None and ``metadata`` contains a ``"vocabulary_path"``
                        key, a warning is printed reminding the caller to load
                        and pass the vocabulary.  If None and no metadata hint
                        exists, the bundled 48k English corpus is used (same
                        as ``create_engine()`` default) - this is correct for
                        engines that were originally built with the default
                        vocabulary, but silently wrong for engines built with
                        a custom vocabulary.

                        Best practice: always store the vocabulary path in
                        ``metadata`` when calling ``to_config()``::

                            config = engine.to_config(
                                preset="production",
                                metadata={"vocabulary_path": str(vocab_path)},
                            )

                        And restore it explicitly::

                            import json
                            config = EngineConfig.from_json(config_text)
                            vocab = json.load(open(config.metadata["vocabulary_path"]))
                            engine = config.build(vocabulary=vocab)

            history:    Pre-loaded ``History`` instance.  If None, starts
                        with an empty in-memory history.

        Returns:
            A fully initialised ``AutocompleteEngine``.
        """
        import warnings

        from aac.presets import create_engine

        if self.preset is not None:
            if vocabulary is None and "vocabulary_path" in self.metadata:
                warnings.warn(
                    f"EngineConfig.build() was called without a vocabulary, but "
                    f"metadata contains 'vocabulary_path': "
                    f"{self.metadata['vocabulary_path']!r}. "
                    f"The engine will use the bundled English corpus instead of "
                    f"the original vocabulary. Pass vocabulary= explicitly if "
                    f"this engine was built with a custom vocabulary.",
                    UserWarning,
                    stacklevel=2,
                )
            return create_engine(self.preset, vocabulary=vocabulary, history=history)

        raise NotImplementedError(
            "Reconstructing a custom engine (preset=None) from config requires "
            "caller-supplied predictor and ranker instances. "
            "Create the engine directly via AutocompleteEngine() or use a named "
            "preset so that build() can delegate to create_engine()."
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
                diffs.append(f"predictor added: {name!r} (weight={b.weight})")  # type: ignore[union-attr]
            elif b is None:
                diffs.append(f"predictor removed: {name!r} (was weight={a.weight})")
            elif abs(a.weight - b.weight) > 1e-9:
                diffs.append(
                    f"predictor {name!r} weight: {a.weight} → {b.weight}"
                )

        self_rankers = [r.name for r in self.rankers]
        other_rankers = [r.name for r in other.rankers]
        if self_rankers != other_rankers:
            diffs.append(f"rankers: {self_rankers} → {other_rankers}")
        else:
            for a, b in zip(self.rankers, other.rankers, strict=False):
                if a.params != b.params:
                    diffs.append(
                        f"ranker {a.name!r} params: {a.params} → {b.params}"
                    )

        return diffs

    def __repr__(self) -> str:
        pred_summary = ", ".join(f"{p.name}×{p.weight}" for p in self.predictors)
        return (
            f"EngineConfig(preset={self.preset!r}, "
            f"predictors=[{pred_summary}], "
            f"rankers={[r.name for r in self.rankers]!r})"
        )
