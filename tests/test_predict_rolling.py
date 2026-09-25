"""Tests for the per-step parallelisation of scripts/predict_rolling.py (P04).

These tests exercise the real _run_one_step()/pd.concat() code path used by both the
sequential and the per-step-parallel loops in predict_rolling.main(), with
execute_train_predict_step() monkeypatched to a cheap deterministic stub (real LightGBM/SVC
training would be too slow for a unit test and is already covered by test_classifiers.py).
"""
import os
import time

import pandas as pd
import pytest
from joblib import Parallel, delayed

import scripts.predict_rolling as predict_rolling
from scripts.predict_rolling import (
    _run_one_step,
    _sort_and_validate_step_results,
    sweep_leftover_temp_model_dirs,
    TMP_MODEL_DIR_PREFIX,
)


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


"""P05: defensive sort/assert of parallel step-result ordering, and temp-dir cleanup."""


def test_sort_and_validate_step_results_sorts_shuffled_results():
    # Arrange: correctly-orderable results submitted/returned out of order.
    step_results = [(2, pd.DataFrame({"x": [2]})), (0, pd.DataFrame({"x": [0]})), (1, pd.DataFrame({"x": [1]}))]

    # Act
    sorted_results = _sort_and_validate_step_results(step_results, prediction_steps=3)

    # Assert
    assert [step for step, _ in sorted_results] == [0, 1, 2]


def test_sort_and_validate_step_results_raises_on_duplicate_step_index():
    # Arrange: step 1 appears twice, step 2 is missing.
    step_results = [(0, pd.DataFrame({"x": [0]})), (1, pd.DataFrame({"x": [1]})), (1, pd.DataFrame({"x": [1]}))]

    # Act / Assert
    with pytest.raises(ValueError, match="unexpected/duplicate/missing step indices"):
        _sort_and_validate_step_results(step_results, prediction_steps=3)


def test_sort_and_validate_step_results_raises_on_missing_step_index():
    # Arrange: only steps 0 and 2 present, step 1 missing (a gap).
    step_results = [(0, pd.DataFrame({"x": [0]})), (2, pd.DataFrame({"x": [2]}))]

    # Act / Assert
    with pytest.raises(ValueError, match="unexpected/duplicate/missing step indices"):
        _sort_and_validate_step_results(step_results, prediction_steps=3)


def test_run_one_step_cleans_up_temp_dir_after_normal_run(monkeypatch):
    """Per-step temp dir must be removed once the worker finishes normally (already covered
    indirectly by test_run_one_step_isolates_model_store above; this asserts it directly via
    the sweep helper's prefix so both tests stay in sync with TMP_MODEL_DIR_PREFIX)."""
    from service.App import App

    def _stub(config, train_df, predict_df, parallel):
        return pd.DataFrame({"label_pred": predict_df["feature_a"]}, index=predict_df.index)

    monkeypatch.setattr(predict_rolling, "execute_train_predict_step", _stub)

    df = _make_df(360)
    train_features_all = ["feature_a", "feature_b"]
    config = {"symbol": "TESTSYM", "data_folder": "/tmp", "model_folder": "models"}

    # Act
    _run_one_step(
        config, df, step=0, prediction_start=200, prediction_size=40,
        label_horizon=5, train_length=0, train_features_all=train_features_all,
        parallel=None, isolate_model_store=True,
    )

    # Assert: nothing matching our prefix is left behind for the sweep to find.
    assert sweep_leftover_temp_model_dirs() == []


def test_run_one_step_cleans_up_temp_dir_when_worker_raises(monkeypatch):
    """try/finally in _run_one_step must remove the worker's own temp dir even when the
    step raises an exception inside execute_train_predict_step (covers normal Python
    exceptions; a hard SIGKILL cannot be caught here and is handled by the parent-side sweep
    instead, see test_sweep_leftover_temp_model_dirs_removes_only_matching_dirs)."""
    captured_model_path = {}

    def _stub_raises(config, train_df, predict_df, parallel):
        from service.App import App
        captured_model_path["path"] = App.model_store.model_path
        raise RuntimeError("simulated failure inside worker step")

    monkeypatch.setattr(predict_rolling, "execute_train_predict_step", _stub_raises)

    df = _make_df(360)
    train_features_all = ["feature_a", "feature_b"]
    config = {"symbol": "TESTSYM", "data_folder": "/tmp", "model_folder": "models"}

    # Act
    with pytest.raises(RuntimeError, match="simulated failure inside worker step"):
        _run_one_step(
            config, df, step=0, prediction_start=200, prediction_size=40,
            label_horizon=5, train_length=0, train_features_all=train_features_all,
            parallel=None, isolate_model_store=True,
        )

    # Assert
    assert not captured_model_path["path"].exists(), "temp model dir must be cleaned up even when the step raises"


def test_sweep_leftover_temp_model_dirs_removes_only_matching_dirs(tmp_path):
    # Arrange: a leftover temp dir from a previous (simulated) hard-killed run, plus an
    # unrelated directory that must be left alone. Simulate staleness by evaluating the
    # sweep at a reference time far in the future of the dirs' mtime.
    leftover_dir = tmp_path / f"{TMP_MODEL_DIR_PREFIX}2_abc123"
    leftover_dir.mkdir()
    (leftover_dir / "some_model.bin").write_text("stale model bytes")

    unrelated_dir = tmp_path / "some_other_unrelated_dir"
    unrelated_dir.mkdir()
    (unrelated_dir / "keep_me.txt").write_text("do not touch")

    stale_reference_time = time.time() + predict_rolling.STALE_TEMP_DIR_AGE_SECONDS + 60

    # Act
    removed = sweep_leftover_temp_model_dirs(base_dir=str(tmp_path), now=stale_reference_time)

    # Assert
    assert removed == [str(leftover_dir)]
    assert not leftover_dir.exists()
    assert unrelated_dir.exists()
    assert (unrelated_dir / "keep_me.txt").exists()


def test_sweep_leftover_temp_model_dirs_spares_fresh_concurrent_run_dir(tmp_path):
    """Regression test for the race condition where sweeping on startup could delete another
    concurrently-running predict_rolling.py process's still-active temp model dir, because both
    match the same name prefix. Only a dir whose mtime is older than
    STALE_TEMP_DIR_AGE_SECONDS (i.e. abandoned by a hard-killed run) must be removed; a dir
    that was touched recently (as a live worker continuously does while executing its step)
    must be left alone."""
    # Arrange: two dirs from two different "runs", both matching the prefix.
    stale_dir = tmp_path / f"{TMP_MODEL_DIR_PREFIX}0_stalerun"
    stale_dir.mkdir()
    (stale_dir / "some_model.bin").write_text("abandoned model bytes")

    fresh_dir = tmp_path / f"{TMP_MODEL_DIR_PREFIX}0_freshrun"
    fresh_dir.mkdir()
    (fresh_dir / "some_model.bin").write_text("actively used model bytes")

    reference_time = time.time()
    stale_mtime = reference_time - predict_rolling.STALE_TEMP_DIR_AGE_SECONDS - 60
    os.utime(stale_dir, (stale_mtime, stale_mtime))
    # fresh_dir keeps its just-created (now-ish) mtime, simulating a live worker still
    # writing to it.

    # Act
    removed = sweep_leftover_temp_model_dirs(base_dir=str(tmp_path), now=reference_time)

    # Assert: only the stale dir from the abandoned run is removed; the concurrently-running
    # process's fresh dir survives untouched.
    assert removed == [str(stale_dir)]
    assert not stale_dir.exists()
    assert fresh_dir.exists()
    assert (fresh_dir / "some_model.bin").exists()
