"""
Contract tests: every Predictor implementation must satisfy the
PredictorContractTestMixin invariants. Add new predictors here.
"""
from __future__ import annotations

from aac.domain.history import History
from aac.domain.types import Predictor
from aac.predictors.adaptive_symspell import AdaptiveSymSpellPredictor
from aac.predictors.edit_distance import EditDistancePredictor
from aac.predictors.frequency import FrequencyPredictor
from aac.predictors.history import HistoryPredictor
from aac.predictors.static_prefix import StaticPrefixPredictor
from aac.predictors.symspell import SymSpellPredictor
from aac.predictors.trie import TriePrefixPredictor
from aac.predictors.trigram import TrigramPredictor
from tests.contracts.predictor_contract import PredictorContractTestMixin

_VOCAB = ["hello", "help", "helium", "hero", "world"]
_FREQUENCIES = {"hello": 100, "help": 80, "helium": 30, "hero": 50, "world": 200}


class TestFrequencyPredictorContract(PredictorContractTestMixin):
    def make_predictor(self) -> Predictor:
        return FrequencyPredictor(_FREQUENCIES)


class TestHistoryPredictorContract(PredictorContractTestMixin):
    def make_predictor(self) -> Predictor:
        history = History()
        history.record("h", "hello")
        return HistoryPredictor(history)


class TestStaticPrefixPredictorContract(PredictorContractTestMixin):
    def make_predictor(self) -> Predictor:
        return StaticPrefixPredictor(_VOCAB)


class TestTriePrefixPredictorContract(PredictorContractTestMixin):
    def make_predictor(self) -> Predictor:
        return TriePrefixPredictor(_VOCAB)


class TestEditDistancePredictorContract(PredictorContractTestMixin):
    """EditDistancePredictor without frequencies - distance-only ranking."""
    def make_predictor(self) -> Predictor:
        return EditDistancePredictor(_VOCAB, max_distance=2)


class TestEditDistancePredictorWithFrequenciesContract(PredictorContractTestMixin):
    """
    EditDistancePredictor WITH frequencies - exercises the frequency multiplier
    path in distance_score(). Separate contract class because the scoring
    formula is different: distance_score(base, dist, freq_score) vs
    distance_score(base, dist, 0.0). Both must satisfy all predictor contracts.
    """
    def make_predictor(self) -> Predictor:
        return EditDistancePredictor(_VOCAB, max_distance=2, frequencies=_FREQUENCIES)


class TestTrigramPredictorContract(PredictorContractTestMixin):
    def make_predictor(self) -> Predictor:
        return TrigramPredictor(_VOCAB, max_distance=2)


class TestSymSpellPredictorContract(PredictorContractTestMixin):
    def make_predictor(self) -> Predictor:
        return SymSpellPredictor(_VOCAB, max_distance=2)


class TestAdaptiveSymSpellPredictorContract(PredictorContractTestMixin):
    """
    AdaptiveSymSpellPredictor was missing from contract tests entirely.
    It delegates to two inner SymSpell instances (tight + full) and must
    satisfy all predictor contracts across both branches.
    The prefix "h" (len=1 < short_prefix_len=4) exercises the tight branch;
    "hell" (len=4 >= short_prefix_len=4) exercises the full branch.
    Both branches are exercised via the contract test's make_predictor().
    """
    def make_predictor(self) -> Predictor:
        return AdaptiveSymSpellPredictor(
            vocabulary=_VOCAB,
            max_distance=2,
            short_prefix_len=4,
            short_max_distance=1,
            frequencies=_FREQUENCIES,
        )
