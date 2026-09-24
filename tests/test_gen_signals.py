"""
Characterization tests for common/gen_signals.py.

These tests pin down the CURRENT behavior of the signal generation building
blocks (generate_combine_scores / combine_scores_relative / combine_scores_difference,
generate_threshold_rule, generate_threshold_rule2, generate_smoothen_scores)
before any experiments are run on the signal-generation pipeline (backlog S01).

They intentionally capture actual current output - including quirky/buggy
edge-case behavior - rather than "corrected" behavior, so they act as a
safety net/regression net for upcoming refactors.
"""
import numpy as np
import pandas as pd
import pytest

from common.gen_signals import (
    generate_smoothen_scores,
    generate_combine_scores,
    combine_scores_relative,
    combine_scores_difference,
    generate_threshold_rule,
    generate_threshold_rule2,
)


# ---------------------------------------------------------------------------
# generate_combine_scores
# ---------------------------------------------------------------------------

def _make_buy_sell_df():
    return pd.DataFrame({
        "buy": [0.1, 0.5, np.nan, 0.9],
        "sell": [0.2, 0.5, 0.3, np.nan],
    })


def test_combine_scores_default_comparison_rule():
    # Arrange
    df = _make_buy_sell_df()
    config = {"columns": ["buy", "sell"], "names": "score"}

    # Act
    out, features = generate_combine_scores(df, config)

    # Assert
    assert features == ["score"]
    expected = pd.Series([-0.2, 0.5, -0.3, np.nan], name="score")
    pd.testing.assert_series_equal(out["score"], expected, check_exact=False)


def test_combine_scores_relative_rule():
    # Arrange
    df = _make_buy_sell_df()
    config = {"columns": ["buy", "sell"], "names": "score", "combine": "relative"}

    # Act
    out, features = generate_combine_scores(df, config)

    # Assert
    assert features == ["score"]
    expected = pd.Series([-1.0 / 3.0, 0.0, np.nan, np.nan], name="score")
    pd.testing.assert_series_equal(out["score"], expected, check_exact=False)


def test_combine_scores_difference_rule():
    # Arrange
    df = _make_buy_sell_df()
    config = {"columns": ["buy", "sell"], "names": "score", "combine": "difference"}

    # Act
    out, features = generate_combine_scores(df, config)

    # Assert
    assert features == ["score"]
    expected = pd.Series([-0.1, 0.0, np.nan, np.nan], name="score")
    pd.testing.assert_series_equal(out["score"], expected, check_exact=False)


def test_combine_scores_difference_with_coefficient_and_constant():
    # Arrange
    df = _make_buy_sell_df()
    config = {
        "columns": ["buy", "sell"], "names": "score", "combine": "difference",
        "coefficient": 2, "constant": 1,
    }

    # Act
    out, features = generate_combine_scores(df, config)

    # Assert: (diff * 2) + 1
    expected = pd.Series([0.8, 1.0, np.nan, np.nan], name="score")
    pd.testing.assert_series_equal(out["score"], expected, check_exact=False)


def test_combine_scores_missing_columns_raises_value_error():
    # Arrange
    df = _make_buy_sell_df()
    config = {"names": "score"}

    # Act / Assert
    with pytest.raises(ValueError):
        generate_combine_scores(df, config)


def test_combine_scores_columns_not_a_two_item_list_raises_value_error():
    # Arrange
    df = _make_buy_sell_df()
    config = {"columns": ["buy"], "names": "score"}

    # Act / Assert
    with pytest.raises(ValueError):
        generate_combine_scores(df, config)


def test_combine_scores_relative_helper_directly():
    # Arrange
    df = _make_buy_sell_df()

    # Act
    result = combine_scores_relative(df, "buy", "sell", "score_out")

    # Assert
    expected = pd.Series([-1.0 / 3.0, 0.0, np.nan, np.nan], name="score_out")
    pd.testing.assert_series_equal(result.rename("score_out"), expected, check_exact=False)
    pd.testing.assert_series_equal(df["score_out"], expected, check_exact=False)


def test_combine_scores_difference_helper_directly():
    # Arrange
    df = _make_buy_sell_df()

    # Act
    result = combine_scores_difference(df, "buy", "sell", "score_out")

    # Assert
    expected = pd.Series([-0.1, 0.0, np.nan, np.nan], name="score_out")
    pd.testing.assert_series_equal(result.rename("score_out"), expected, check_exact=False)
    pd.testing.assert_series_equal(df["score_out"], expected, check_exact=False)


# ---------------------------------------------------------------------------
# generate_threshold_rule
# ---------------------------------------------------------------------------

def _threshold_config(columns, buy_threshold=0.5, sell_threshold=-0.5):
    return {
        "columns": columns,
        "names": ["buy_sig", "sell_sig"],
        "parameters": {
            "buy_signal_threshold": buy_threshold,
            "sell_signal_threshold": sell_threshold,
        },
    }


def test_threshold_rule_string_column_boundary_values():
    # Arrange: includes values exactly at the boundary and NaN
    df = pd.DataFrame({"score": [0.1, 0.5, np.nan, -0.5]})
    config = _threshold_config("score")

    # Act
    out, features = generate_threshold_rule(df, config)

    # Assert
    assert features == ["buy_sig", "sell_sig"]
    # 0.5 >= 0.5 -> True (buy threshold is inclusive)
    assert list(out["buy_sig"]) == [False, True, False, False]
    # -0.5 <= -0.5 -> True (sell threshold is inclusive); NaN comparisons are False
    assert list(out["sell_sig"]) == [False, False, False, True]


def test_threshold_rule_empty_dataframe():
    # Arrange
    df = pd.DataFrame({"score": pd.Series([], dtype=float)})
    config = _threshold_config("score")

    # Act
    out, features = generate_threshold_rule(df, config)

    # Assert
    assert features == ["buy_sig", "sell_sig"]
    assert len(out) == 0
    assert list(out.columns) == ["score", "buy_sig", "sell_sig"]


def test_threshold_rule_list_column_currently_raises_key_error():
    """
    Current (buggy) behavior: when 'columns' is passed as a list, the function
    wraps it in another list (`columns = [columns]`), producing a nested list
    that pandas cannot resolve to a column, raising a KeyError. This test pins
    down the CURRENT behavior so any future fix is a deliberate, visible change.
    """
    # Arrange
    df = pd.DataFrame({"score": [0.1, 0.5, np.nan, -0.5]})
    config = _threshold_config(["score"])

    # Act / Assert
    with pytest.raises(KeyError):
        generate_threshold_rule(df, config)


def test_threshold_rule_missing_columns_raises_value_error():
    # Arrange
    df = pd.DataFrame({"score": [0.1, 0.5]})
    config = {"names": ["buy_sig", "sell_sig"], "parameters": {}}

    # Act / Assert
    with pytest.raises(ValueError):
        generate_threshold_rule(df, config)


# ---------------------------------------------------------------------------
# generate_threshold_rule2
# ---------------------------------------------------------------------------

def test_threshold_rule2_requires_both_scores_above_thresholds():
    # Arrange
    df = pd.DataFrame({"s1": [0.6, 0.4, np.nan], "s2": [0.6, 0.6, 0.6]})
    config = {
        "columns": ["s1", "s2"],
        "names": ["buy_sig", "sell_sig"],
        "parameters": {
            "buy_signal_threshold": 0.5, "buy_signal_threshold_2": 0.5,
            "sell_signal_threshold": -0.5, "sell_signal_threshold_2": -0.5,
        },
    }

    # Act
    out, features = generate_threshold_rule2(df, config)

    # Assert
    assert features == ["buy_sig", "sell_sig"]
    assert list(out["buy_sig"]) == [True, False, False]
    assert list(out["sell_sig"]) == [False, False, False]


def test_threshold_rule2_invalid_columns_raises_value_error():
    # Arrange
    df = pd.DataFrame({"s1": [0.6]})
    config = {"columns": "s1", "names": ["buy_sig", "sell_sig"], "parameters": {}}

    # Act / Assert
    with pytest.raises(ValueError):
        generate_threshold_rule2(df, config)


# ---------------------------------------------------------------------------
# generate_smoothen_scores
# ---------------------------------------------------------------------------

def test_smoothen_scores_integer_window_rolling_mean():
    # Arrange
    df = pd.DataFrame({"a": [0.1, 0.2, 0.3, np.nan, 0.5], "b": [0.2, 0.3, 0.4, 0.5, np.nan]})
    config = {"columns": ["a", "b"], "names": "smooth", "window": 3}

    # Act
    out, features = generate_smoothen_scores(df, config)

    # Assert
    assert features == ["smooth"]
    expected = pd.Series(
        [0.15, 0.2, 0.25, 0.366666666667, 0.45], name="smooth"
    )
    pd.testing.assert_series_equal(out["smooth"], expected, check_exact=False)


def test_smoothen_scores_point_threshold_binarizes_output():
    # Arrange
    df = pd.DataFrame({"a": [0.1, 0.2, 0.3, np.nan, 0.5], "b": [0.2, 0.3, 0.4, 0.5, np.nan]})
    config = {"columns": ["a", "b"], "names": "smooth", "point_threshold": 0.3}

    # Act
    out, features = generate_smoothen_scores(df, config)

    # Assert: mean(row) >= 0.3 (boundary case at row index 2: mean == 0.3 -> True)
    assert list(out["smooth"]) == [False, False, True, True, True]


def test_smoothen_scores_missing_columns_raises_value_error():
    # Arrange
    df = pd.DataFrame({"a": [0.1, 0.2]})
    config = {"names": "smooth"}

    # Act / Assert
    with pytest.raises(ValueError):
        generate_smoothen_scores(df, config)


def test_smoothen_scores_missing_names_raises_value_error():
    # Arrange
    df = pd.DataFrame({"a": [0.1, 0.2]})
    config = {"columns": ["a"]}

    # Act / Assert
    with pytest.raises(ValueError):
        generate_smoothen_scores(df, config)


# ---------------------------------------------------------------------------
# Offline vs "online" (last_rows) consistency and no-look-ahead checks
# ---------------------------------------------------------------------------

def test_combine_scores_no_look_ahead():
    """
    Appending future rows must not change previously computed values for
    row-wise (non-rolling) transformations such as generate_combine_scores.
    """
    # Arrange
    df_full = pd.DataFrame({
        "buy": [0.1, 0.5, 0.3, 0.9, 0.2],
        "sell": [0.2, 0.5, 0.6, 0.1, 0.4],
    })
    df_partial = df_full.iloc[:3].copy()
    config = {"columns": ["buy", "sell"], "names": "score", "combine": "difference"}

    # Act
    out_full, _ = generate_combine_scores(df_full.copy(), config)
    out_partial, _ = generate_combine_scores(df_partial, config)

    # Assert: first 3 rows identical regardless of future rows being present
    pd.testing.assert_series_equal(
        out_full["score"].iloc[:3].reset_index(drop=True),
        out_partial["score"].reset_index(drop=True),
    )


def test_threshold_rule_offline_vs_online_last_rows_equal():
    """
    generate_threshold_rule is row-wise, so running it on the full history and
    then taking the tail must equal running it directly on only the last N
    rows (simulating the online 'last_rows' mode).
    """
    # Arrange
    df_full = pd.DataFrame({"score": [0.1, 0.6, -0.6, 0.2, 0.55, -0.9]})
    config = _threshold_config("score")
    last_rows = 2
    df_online = df_full.tail(last_rows).reset_index(drop=True).copy()

    # Act
    out_full, _ = generate_threshold_rule(df_full.copy(), config)
    out_online, _ = generate_threshold_rule(df_online, config)

    # Assert
    tail_full = out_full[["buy_sig", "sell_sig"]].tail(last_rows).reset_index(drop=True)
    tail_online = out_online[["buy_sig", "sell_sig"]]
    pd.testing.assert_frame_equal(tail_full, tail_online)
