"""
adaptive-autocomplete: a composable autocomplete engine built from scratch.

Zero required dependencies. Pluggable predictors, explainable ranking,
offline evaluation, and automated weight optimisation.

Quick start::

    from aac.presets import create_engine

    engine = create_engine("production")
    engine.suggest("programing")          # ['programming']  (typo recovered)
    engine.record_selection("prog", "programming")
    engine.explain("prog")                # per-suggestion score breakdown

See https://github.com/bonnie-mcconnell/adaptive-autocomplete for full docs.
"""

from aac.domain.thread_safe_history import ThreadSafeHistory

__version__ = "1.0.0"
__author__ = "Bonnie McConnell"
__license__ = "MIT"
__all__ = ["__version__", "__author__", "__license__", "ThreadSafeHistory"]
