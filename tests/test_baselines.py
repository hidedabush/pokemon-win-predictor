import numpy as np
import pandas as pd
import pytest

from experiments import run_baselines
from src.evaluation.metrics import evaluate
from src.models.baselines import BASELINES


@pytest.fixture
def X():
    return pd.DataFrame({"stats_bst": [10, -5, 0],
                         "match_cover": [-1, 0, 3],
                         "rating_diff": [0, 20, -15]})


def test_coin_flip(X):
    np.testing.assert_array_equal(BASELINES["coin_flip"](X), [0.5] * 3)


@pytest.mark.parametrize("name, expected", [
    ("higher_bst", [0.6, 0.4, 0.5]),
    ("more_type_adv", [0.4, 0.5, 0.6]),
    ("higher_rating", [0.5, 0.6, 0.4]),
])
def test_rules(X, name, expected):
    np.testing.assert_array_equal(BASELINES[name](X), expected)


@pytest.mark.parametrize("name", list(BASELINES))
def test_swap(X, name):
    predict = BASELINES[name]
    np.testing.assert_allclose(predict(-X), 1 - predict(X))


def test_coin_metrics(X):
    scores = evaluate([0, 1, 1], BASELINES["coin_flip"](X))
    assert scores["log_loss"] == pytest.approx(np.log(2))
    assert scores["brier"] == pytest.approx(0.25)
    assert scores["accuracy"] == pytest.approx(2 / 3)


def test_runner(X, monkeypatch, tmp_path):
    calls = []
    def load(split):
        calls.append(split)
        return X, pd.Series([0, 1, 1])
    monkeypatch.setattr(run_baselines, "load_features", load)
    monkeypatch.setattr(run_baselines, "ROOT", tmp_path)
    monkeypatch.setattr(run_baselines, "load_config", lambda: {"paths": {"results_dir": "results"}})
    results = run_baselines.main()
    assert calls == ["val"]
    assert len(results) == 4
    calls.clear()
    results = run_baselines.main(include_test=True)
    assert calls == ["val", "test"]
    assert len(results) == 8
    saved = pd.read_csv(tmp_path / "results/baselines.csv")
    assert list(saved.columns) == ["model", "split", "accuracy", "log_loss", "brier"]
    assert len(saved) == 8
