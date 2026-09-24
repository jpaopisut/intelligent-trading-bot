"""Tests for the per-step parallelisation of scripts/predict_rolling.py (P04).

These tests exercise the real _run_one_step()/pd.concat() code path used by both the
sequential and the per-step-parallel loops in predict_rolling.main(), with
execute_train_predict_step() monkeypatched to a cheap deterministic stub (real LightGBM/SVC
training would be too slow for a unit test and is already covered by test_classifiers.py).
"""
import time

import pandas as pd
import pytest
from joblib import Parallel, delayed

import scripts.predict_rolling as predict_rolling
from scripts.predict_rolling import _run_one_step


def _make_df(n_rows: int = 360) -> pd.DataFrame:
    """Small synthetic feature matrix: a couple of numeric features + one binary label."""
    return pd.DataFrame({
        "close": [100.0 + i * 0.1 for i in range(n_rows)],
        "feature_a": [float(i % 7) for i in range(n_rows)],
        "feature_b": [float(i % 5) - 2.0 for i in range(n_rows)],
        "label": [i % 2 for i in range(n_rows)],
    })


def _stub_execute_train_predict_step(sleep_by_step=None):
    """Build a stub for execute_train_predict_step that is cheap and deterministic.

    The returned prediction depends only on train_df/predict_df content (mean of the
    training feature plus the predicted row's own feature value), so any bug in the
    train/predict slicing done by _run_one_step would be visible as a value mismatch.
    """
    def _stub(config, train_df, predict_df, parallel):
        if sleep_by_step is not None:
            step = config["_test_step_hint"]
            time.sleep(sleep_by_step(step))
        train_mean = train_df["feature_a"].mean() if len(train_df) else 0.0
        predicted = predict_df["feature_a"] + train_mean
        return pd.DataFrame({"label_pred": predicted}, index=predict_df.index)
    return _stub


def _run_sequential(config, df, prediction_start, prediction_size, prediction_steps,
                     label_horizon, train_length, train_features_all):
    labels_hat_df = pd.DataFrame()
    for step in range(prediction_steps):
        _, predict_labels_df = _run_one_step(
            config, df, step, prediction_start, prediction_size,
            label_horizon, train_length, train_features_all,
            parallel=None,
        )
        labels_hat_df = pd.concat([labels_hat_df, predict_labels_df])
    return labels_hat_df


def _run_parallel(config, df, prediction_start, prediction_size, prediction_steps,
                   label_horizon, train_length, train_features_all):
    # Use the "threading" backend (fast, in-process) since the stub does not touch
    # App.model_store; isolate_model_store=False is therefore safe here. The real
    # isolate_model_store=True path (used with backend="loky" in production) is
    # covered separately in test_run_one_step_isolates_model_store below.
    parallel = Parallel(n_jobs=3, backend="threading")
    step_results = parallel(
        delayed(_run_one_step)(
            config, df, step, prediction_start, prediction_size,
            label_horizon, train_length, train_features_all,
        )
        for step in range(prediction_steps)
    )
    labels_hat_df = pd.DataFrame()
    for step, predict_labels_df in step_results:
        labels_hat_df = pd.concat([labels_hat_df, predict_labels_df])
    return labels_hat_df, step_results


def test_parallel_and_sequential_step_loops_produce_identical_output(monkeypatch):
    # Arrange
    monkeypatch.setattr(
        predict_rolling, "execute_train_predict_step", _stub_execute_train_predict_step()
    )
    df = _make_df(360)
    train_features_all = ["feature_a", "feature_b"]
    config = {"_test_step_hint": 0}
    prediction_start, prediction_size, prediction_steps = 200, 40, 3
    label_horizon, train_length = 5, 0

    # Act
    sequential_df = _run_sequential(
        config, df, prediction_start, prediction_size, prediction_steps,
        label_horizon, train_length, train_features_all,
    )
    parallel_df, _ = _run_parallel(
        config, df, prediction_start, prediction_size, prediction_steps,
        label_horizon, train_length, train_features_all,
    )

    # Assert
    pd.testing.assert_frame_equal(sequential_df, parallel_df)


def test_parallel_step_loop_preserves_step_order_despite_out_of_order_completion(monkeypatch):
    # Arrange: step 0 sleeps longest, step 2 shortest, so workers finish in reverse order.
    monkeypatch.setattr(
        predict_rolling,
        "execute_train_predict_step",
        _stub_execute_train_predict_step(sleep_by_step=lambda step: 0.05 * (3 - step)),
    )
    df = _make_df(360)
    train_features_all = ["feature_a", "feature_b"]
    prediction_start, prediction_size, prediction_steps = 200, 40, 3
    label_horizon, train_length = 5, 0
    config = {"_test_step_hint": 0}

    # Act
    parallel_df, step_results = _run_parallel(
        config, df, prediction_start, prediction_size, prediction_steps,
        label_horizon, train_length, train_features_all,
    )
    sequential_df = _run_sequential(
        {"_test_step_hint": 0}, df, prediction_start, prediction_size, prediction_steps,
        label_horizon, train_length, train_features_all,
    )

    # Assert: joblib.Parallel returns results in submission order, not completion order.
    assert [step for step, _ in step_results] == [0, 1, 2]
    pd.testing.assert_frame_equal(sequential_df, parallel_df)


def test_run_one_step_isolates_model_store(monkeypatch):
    """isolate_model_store=True must create a private ModelStore backed by a temp dir
    (so concurrent worker processes never read/write the same model files), and clean it
    up again afterwards."""
    from service.App import App

    seen_model_paths = []

    def _stub(config, train_df, predict_df, parallel):
        seen_model_paths.append(App.model_store.model_path)
        return pd.DataFrame({"label_pred": predict_df["feature_a"]}, index=predict_df.index)

    monkeypatch.setattr(predict_rolling, "execute_train_predict_step", _stub)

    df = _make_df(360)
    train_features_all = ["feature_a", "feature_b"]
    config = {"symbol": "TESTSYM", "data_folder": "/tmp", "model_folder": "models"}

    # Act
    step, predict_labels_df = _run_one_step(
        config, df, step=0, prediction_start=200, prediction_size=40,
        label_horizon=5, train_length=0, train_features_all=train_features_all,
        parallel=None, isolate_model_store=True,
    )

    # Assert
    assert len(seen_model_paths) == 1
    used_model_path = seen_model_paths[0]
    assert "itb_rolling_predict_step_0_" in str(used_model_path)
    assert not used_model_path.exists(), "temp model dir must be cleaned up after the step"
    assert len(predict_labels_df) == 40
