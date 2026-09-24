"""Tests for the evaluate_signals backtest validation script."""
import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".claude"
    / "skills"
    / "itb-backtest-validation"
    / "scripts"
    / "evaluate_signals.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_signals", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_signals"] = module
    spec.loader.exec_module(module)
    return module


evaluate_signals = _load_module()


def _make_signals_csv(path: Path) -> None:
    # 10 hourly bars, tz-aware ISO8601 UTC timestamps, alternating buy/sell signals.
    times = pd.date_range("2025-01-01T00:00:00Z", periods=10, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": times.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),
            "open": [100.0 + i for i in range(10)],
            "close": [101.0 + i for i in range(10)],
            "buy_signal_column": [i % 4 == 0 for i in range(10)],
            "sell_signal_column": [i % 4 == 2 for i in range(10)],
        }
    )
    df.to_csv(path, index=False)


def _build_args(file_path: Path, start: str, end: str, holdout_start: str) -> argparse.Namespace:
    parser = evaluate_signals.build_parser()
    return parser.parse_args(
        [
            "--file",
            str(file_path),
            "--start",
            start,
            "--end",
            end,
            "--holdout-start",
            holdout_start,
        ]
    )


def test_run_handles_tz_aware_column_with_naive_cli_timestamps(tmp_path):
    # Arrange
    csv_path = tmp_path / "signals.csv"
    _make_signals_csv(csv_path)
    # Rows 0..9 span 2025-01-01T00:00Z .. 2025-01-01T09:00Z.
    # --start/--end keep rows 1..8 (8 rows), --holdout-start splits at hour 5:
    # TUNING = rows 1..4 (4 rows), HOLDOUT = rows 5..8 (4 rows).
    args = _build_args(
        csv_path,
        start="2025-01-01T01:00:00",
        end="2025-01-01T09:00:00",
        holdout_start="2025-01-01T05:00:00",
    )

    # Act
    results = evaluate_signals.run(args)

    # Assert
    assert set(results.keys()) == {"ALL", "TUNING", "HOLDOUT"}
    assert results["ALL"]["bars"] == 8
    assert results["TUNING"]["bars"] == 4
    assert results["HOLDOUT"]["bars"] == 4


def test_load_frame_produces_tz_aware_timestamp_column(tmp_path):
    # Arrange
    csv_path = tmp_path / "signals.csv"
    _make_signals_csv(csv_path)

    # Act
    df = evaluate_signals.load_frame(csv_path, "timestamp")

    # Assert
    assert df["timestamp"].dtype.tz is not None


def test_nan_price_rows_do_not_corrupt_equity_metrics(tmp_path):
    # Arrange: exchange-outage rows have NaN prices (as produced by scripts.merge raster filling)
    times = pd.date_range("2025-01-01T00:00:00Z", periods=8, freq="h")
    df = pd.DataFrame({
        "timestamp": times.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),
        "open":  [100.0, 100.0, 102.0, float("nan"), float("nan"), 104.0, 103.0, 103.0],
        "close": [100.0, 102.0, 103.0, float("nan"), float("nan"), 103.0, 103.0, 103.0],
        "buy_signal_column":  [True, False, False, False, False, False, False, False],
        "sell_signal_column": [False, False, False, False, True, False, False, False],
    })
    file = tmp_path / "signals.csv"
    df.to_csv(file, index=False)
    args = evaluate_signals.build_parser().parse_args(["--file", str(file), "--fee", "0", "--slippage", "0"])

    # Act
    result = evaluate_signals.run(args)["ALL"]

    # Assert: metrics are finite; the sell signal on a NaN bar executes at the next valid open (104)
    assert result["max_drawdown_%"] == result["max_drawdown_%"]  # not NaN
    assert result["sharpe"] == result["sharpe"] and result["sharpe"] != 0.0
    assert result["trades"] == 1
    assert result["net_return_%"] == pytest.approx(round(100 * (104 / 100 - 1), 2))
