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
from src.evaluation.metrics import bootstrap_ci, evaluate
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


def test_bootstrap_ci():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    good = np.clip(np.where(y == 1, 0.65, 0.35) + rng.normal(0, 0.1, n), 0.01, 0.99)
    coin = np.full(n, 0.5)
    ci = bootstrap_ci(y, {"good": good, "coin": coin, "same": good.copy()}, reference="good",
                      n_boot=500, seed=0)
    for name, p in {"good": good, "coin": coin}.items():
        m = evaluate(y, p)
        for metric in ("accuracy", "log_loss", "brier"):
            lo, hi = ci[name][metric]
            assert lo <= m[metric] <= hi
    # Accuracy interval width matches the textbook standard error (about 1.96 * sqrt(pq/n) per side).
    lo, hi = ci["good"]["accuracy"]
    acc = evaluate(y, good)["accuracy"]
    assert abs((hi - lo) / 2 - 1.96 * np.sqrt(acc * (1 - acc) / n)) < 0.004
    # Paired difference: point estimate equals the plain difference; a clearly worse model is caught,
    # and a copy of the reference gives exactly zero.
    d, dlo, dhi = ci["coin"]["diff_log_loss"]
    assert np.isclose(d, evaluate(y, coin)["log_loss"] - evaluate(y, good)["log_loss"])
    assert 0 < dlo <= d <= dhi
    assert ci["same"]["diff_accuracy"] == (0.0, 0.0, 0.0)
    assert "diff_accuracy" not in ci["good"]


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
    cfg = {**cfg, "models": {**cfg["models"],  # small grids so the test is quick
                             "random_forest": {"n_estimators": 20, "max_depth_grid": [3, None]},
                             "gradient_boosting": {"learning_rate": 0.1, "max_depth_grid": [2],
                                                   "max_iter": 20}}}
    processed, figures = tmp / "processed_dir", tmp_path / "figures"
    table = run_main.main(cfg=cfg, processed_dir=processed, results_dir=tmp_path,
                          figures_dir=figures)

    assert sorted(table["model"].unique()) == sorted(run_main.DEFAULT_MODELS)
    assert set(table["setting"]) == set(run_main.SETTINGS)
    assert table.groupby("setting").size().eq(len(run_main.DEFAULT_MODELS)).all()
    gb = table[table["model"] == "gradient_boosting"]["params"]
    assert gb.str.contains("rounds=").all()  # tuned number of boosting rounds is recorded
    for name in ("calibration_test", "hyperparam_logistic", "feature_importance_logistic"):
        assert (figures / f"{name}.png").exists()

    # Test is scored once: a second run refuses unless forced.
    with pytest.raises(SystemExit):
        run_main.main(cfg=cfg, processed_dir=processed, results_dir=tmp_path, figures_dir=figures)
    for col in ("accuracy", "log_loss", "brier", "val_log_loss"):
        assert table[col].between(0, 10).all()
    for m in ("accuracy", "log_loss", "brier"):  # every number sits inside its own interval
        assert (table[f"{m}_lo"] <= table[m]).all() and (table[m] <= table[f"{m}_hi"]).all()
    is_ref = table["model"] == run_main.REFERENCE
    assert table.loc[is_ref, "log_loss_vs_logistic"].isna().all()
    others = table[~is_ref]
    assert others["log_loss_vs_logistic"].notna().all()
    assert (others["log_loss_vs_logistic_lo"] <= others["log_loss_vs_logistic"]).all()
    assert (others["log_loss_vs_logistic"] <= others["log_loss_vs_logistic_hi"]).all()

    n_test = len(pd.read_parquet(tmp / "interim_dir/test.parquet"))
    preds = pd.read_csv(tmp_path / "test_predictions.csv")
    assert list(preds.columns) == ["battle_id", "model", "setting", "p_win", "y"]
    assert len(preds) == n_test * len(table)
    assert preds["p_win"].between(0, 1).all()

    # Predictions line up with the test labels by battle_id.
    _, y_test = load_features("test", processed_dir=processed)
    for (model, setting), group in preds.groupby(["model", "setting"]):
        assert group["y"].tolist() == y_test.loc[group["battle_id"]].tolist()

    curve = pd.read_csv(tmp_path / "logistic_c_curve.csv")
    assert len(curve) == 2 * len(cfg["models"]["logistic"]["C_grid"])
    assert curve.groupby("setting")["best"].sum().eq(1).all()
    coefs = pd.read_csv(tmp_path / "logistic_top_coefficients.csv")
    assert not coefs[coefs["setting"] == "teams only"]["feature"].str.startswith("rating_").any()
    assert coefs[coefs["setting"] == "teams + ratings"]["feature"].notna().all()


def test_analyze_errors_reads_run_main_output(fake_features, tmp_path):
    # Andy's error analysis must accept the prediction file run_main writes.
    from experiments.analyze_errors import analyze_predictions
    cfg, tmp = fake_features
    run_main.main(models=["logistic"], cfg=cfg, processed_dir=tmp / "processed_dir",
                  results_dir=tmp_path, figures_dir=tmp_path / "figures")
    preds = pd.read_csv(tmp_path / "test_predictions.csv", dtype={"battle_id": str})
    summary, examples = analyze_predictions(preds, pd.read_parquet(tmp / "interim_dir/test.parquet"))
    assert len(summary) == 2 and summary["n"].eq(len(preds) // 2).all()
