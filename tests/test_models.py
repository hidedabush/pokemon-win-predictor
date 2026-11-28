"""Tests for src/models/logistic.py and experiments/run_main.py.

The run_main test builds features from fake battles in a temp folder (like test_pipeline.py),
so it never reads or writes the real data/ or results/ folders.
"""

import numpy as np
import pandas as pd
import pytest

from experiments import run_main
from src.data.parse_logs import build_battles
from src.data.split import split
from src.features.build_features import build, load_features
from src.models.logistic import make_logistic, top_coefficients, train_logistic
from tests.fake_data import random_battles, tmp_config, write_raw


def planted(n, seed):
    """p1 - p2 features where stats_a and species_x really matter and stats_noise doesn't."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "stats_a": rng.normal(0, 50, n),          # big scale: must be standardized
        "stats_noise": rng.normal(0, 1, n),
        "species_x": rng.choice([-1, 0, 1], n),
        "species_y": rng.choice([-1, 0, 1], n),
    })
    logit = 2.0 * X["stats_a"] / 50 + 0.7 * X["species_x"]  # stats_a: 2.0 per std
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_picks_lowest_val_loss_and_learns_signal():
    X_tr, y_tr = planted(4000, 0)
    X_va, y_va = planted(2000, 1)
    grid = [1e-4, 1e-2, 1.0]
    model, curve = train_logistic(X_tr, y_tr, X_va, y_va, grid)
    assert curve["C"].tolist() == grid
    best = curve.loc[curve["val_log_loss"].idxmin()]
    assert model.named_steps["clf"].C == best["C"] and curve["best"].sum() == 1
    assert best["val_accuracy"] > 0.65  # the planted signal is found
    coef = top_coefficients(model, X_tr.columns, k=2)
    assert coef["feature"].iloc[0] == "stats_a"  # biggest effect per std
    assert coef["coef"].is_monotonic_decreasing and coef["feature"].is_unique


def test_species_columns_are_not_scaled_and_scaler_uses_train_only():
    X_tr, y_tr = planted(500, 0)
    model = make_logistic(X_tr.columns, 1.0).fit(X_tr, y_tr)
    out = pd.DataFrame(model.named_steps["prep"].transform(X_tr),
                       columns=model.named_steps["prep"].get_feature_names_out())
    assert set(out["species_x"]) <= {-1, 0, 1}
    assert abs(out["stats_a"].mean()) < 1e-9 and abs(out["stats_a"].std(ddof=0) - 1) < 1e-9
    # Transforming other data must reuse the train mean/std, not refit.
    X_va, _ = planted(500, 1)
    scaler = model.named_steps["prep"].named_transformers_["scale"]
    assert np.allclose(scaler.mean_, X_tr[["stats_a", "stats_noise"]].mean())
    assert not np.allclose(scaler.mean_, X_va[["stats_a", "stats_noise"]].mean())


def test_swapping_teams_flips_the_probability():
    # Swapped battle = negated features, so P(p1 wins) must become 1 - P (no intercept).
    X_tr, y_tr = planted(2000, 0)
    X_aug = pd.concat([X_tr, -X_tr], ignore_index=True)
    y_aug = pd.concat([y_tr, 1 - y_tr], ignore_index=True)
    model = make_logistic(X_tr.columns, 1.0).fit(X_aug, y_aug)
    X_va, _ = planted(300, 1)
    p, p_swap = model.predict_proba(X_va)[:, 1], model.predict_proba(-X_va)[:, 1]
    assert np.allclose(p + p_swap, 1.0)


def test_mismatched_columns_raise():
    X_tr, y_tr = planted(100, 0)
    with pytest.raises(ValueError):
        train_logistic(X_tr, y_tr, X_tr.drop(columns="species_y"), y_tr, [1.0])


@pytest.fixture(scope="module")
def fake_features(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    cfg = tmp_config(tmp)
    write_raw(cfg, random_battles(300))
    build_battles(cfg)
    split(cfg)
    build(cfg)
    return cfg, tmp


def test_run_main_end_to_end_on_fake_features(fake_features, tmp_path):
    cfg, tmp = fake_features
    processed = tmp / "processed_dir"
    table = run_main.main(cfg=cfg, processed_dir=processed, results_dir=tmp_path)

    # logistic always runs; models that aren't written yet are skipped, not crashed on.
    assert "logistic" in set(table["model"])
    assert set(table["setting"]) == set(run_main.SETTINGS)
    for col in ("accuracy", "log_loss", "brier", "val_log_loss"):
        assert table[col].between(0, 10).all()

    n_test = len(pd.read_parquet(tmp / "interim_dir/test.parquet"))
    preds = pd.read_csv(tmp_path / "test_predictions.csv")
    assert list(preds.columns) == ["battle_id", "model", "setting", "p_win", "y"]
    assert len(preds) == n_test * len(table)
    assert preds["p_win"].between(0, 1).all()

    # Predictions line up with the test labels by battle_id.
    _, y_test = load_features("test", processed_dir=processed)
    first = preds[(preds["model"] == "logistic") & (preds["setting"] == "teams_only")]
    assert first["y"].tolist() == y_test.loc[first["battle_id"]].tolist()

    curve = pd.read_csv(tmp_path / "logistic_c_curve.csv")
    assert len(curve) == 2 * len(cfg["models"]["logistic"]["C_grid"])
    assert curve.groupby("setting")["best"].sum().eq(1).all()
    coefs = pd.read_csv(tmp_path / "logistic_top_coefficients.csv")
    assert not coefs[coefs["setting"] == "teams_only"]["feature"].str.startswith("rating_").any()
    assert coefs[coefs["setting"] == "teams_ratings"]["feature"].notna().all()
