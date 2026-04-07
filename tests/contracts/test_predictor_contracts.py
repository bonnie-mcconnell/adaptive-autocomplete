"""
Contract tests: every Predictor implementation must satisfy the
PredictorContractTestMixin invariants. Add new predictors here.
"""
from __future__ import annotations

from aac.domain.history import History
from aac.predictors.edit_distance import EditDistancePredictor
from aac.predictors.frequency import FrequencyPredictor
from aac.predictors.history import HistoryPredictor
from aac.predictors.static_prefix import StaticPrefixPredictor
from aac.predictors.trie import TriePrefixPredictor
from tests.contracts.predictor_contract import PredictorContractTestMixin

_VOCAB = ["hello", "help", "helium", "hero", "world"]
_FREQUENCIES = {"hello": 100, "help": 80, "helium": 30, "hero": 50, "world": 200}


class TestFrequencyPredictorContract(PredictorContractTestMixin):
    def make_predictor(self) -> FrequencyPredictor:
        return FrequencyPredictor(_FREQUENCIES)


class TestHistoryPredictorContract(PredictorContractTestMixin):
    def make_predictor(self) -> HistoryPredictor:
        history = History()
        history.record("h", "hello")
        return HistoryPredictor(history)


class TestStaticPrefixPredictorContract(PredictorContractTestMixin):
    def make_predictor(self) -> StaticPrefixPredictor:
        return StaticPrefixPredictor(_VOCAB)


class TestTriePrefixPredictorContract(PredictorContractTestMixin):
    def make_predictor(self) -> TriePrefixPredictor:
        return TriePrefixPredictor(_VOCAB)


class TestEditDistancePredictorContract(PredictorContractTestMixin):
    def make_predictor(self) -> EditDistancePredictor:
        return EditDistancePredictor(_VOCAB, max_distance=2)