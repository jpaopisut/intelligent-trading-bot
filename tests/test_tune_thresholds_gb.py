import pytest

from loop.scripts.tune_thresholds_gb import passes_bar, MIN_TRADES, MIN_PF, MAX_EXPOSURE


def _row(trades=50, profit_factor=1.5, exposure=40.0):
    return {"trades": trades, "profit_factor": profit_factor, "exposure_%": exposure}


def test_row_with_too_few_trades_is_excluded():
    # Arrange
    row = _row(trades=MIN_TRADES - 1)

    # Act
    result = passes_bar(row)

    # Assert
    assert result is False


def test_row_with_low_profit_factor_is_excluded():
    # Arrange
    row = _row(profit_factor=MIN_PF - 0.01)

    # Act
    result = passes_bar(row)

    # Assert
    assert result is False


def test_row_with_high_exposure_is_excluded_even_if_sharpe_is_highest():
    # Arrange: high exposure row, otherwise ideal on trades/pf, would have the highest sharpe.
    row = _row(exposure=MAX_EXPOSURE + 26.6)  # e.g. the 86.6% exposure point from E03
    row["sharpe"] = 99.0

    # Act
    result = passes_bar(row)

    # Assert
    assert result is False


def test_normal_passing_row_is_included():
    # Arrange
    row = _row(trades=MIN_TRADES, profit_factor=MIN_PF, exposure=MAX_EXPOSURE)

    # Act
    result = passes_bar(row)

    # Assert
    assert result is True
