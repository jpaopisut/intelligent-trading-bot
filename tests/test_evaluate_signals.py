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


if not SCRIPT_PATH.exists():  # .claude/ tooling is not part of every checkout
    pytest.skip(f"{SCRIPT_PATH} not present", allow_module_level=True)

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


def test_core_weight_zero_is_identical_to_default_behaviour(tmp_path):
    # Arrange
    csv_path = tmp_path / "signals.csv"
    _make_signals_csv(csv_path)
    base_args = evaluate_signals.build_parser().parse_args(["--file", str(csv_path)])
    zero_args = evaluate_signals.build_parser().parse_args(["--file", str(csv_path), "--core-weight", "0"])

    # Act
    baseline = evaluate_signals.run(base_args)["ALL"]
    with_zero_weight = evaluate_signals.run(zero_args)["ALL"]

    # Assert: bit-identical (no core sleeve is even built when core_weight == 0.0)
    assert with_zero_weight == baseline


def test_core_weight_one_equals_buy_and_hold_with_no_strategy_effect(tmp_path):
    # Arrange: a signals file whose strategy sleeve trades a lot and loses money, to prove
    # that at core_weight=1 none of that strategy behaviour leaks into the blended metrics.
    times = pd.date_range("2025-01-01T00:00:00Z", periods=10, freq="h")
    df = pd.DataFrame({
        "timestamp": times.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),
        "open": [100.0 + i for i in range(10)],
        "close": [100.5 + i for i in range(10)],
        "buy_signal_column": [i % 2 == 0 for i in range(10)],  # whipsaw every bar
        "sell_signal_column": [i % 2 == 1 for i in range(10)],
    })
    file = tmp_path / "signals.csv"
    df.to_csv(file, index=False)
    args = evaluate_signals.build_parser().parse_args(
        ["--file", str(file), "--core-weight", "1", "--fee", "0.01", "--slippage", "0.01"]
    )

    # Act
    result = evaluate_signals.run(args)["ALL"]

    # Assert
    assert result["net_return_%"] == result["buy_hold_%"]
    assert result["max_drawdown_%"] == 0.0  # a monotonically rising price never draws down
    assert result["exposure_%"] == 100.0  # core sleeve is always "exposed" at w=1
    # Non-equity trade stats still reflect the (unblended) strategy sleeve.
    assert result["trades"] > 0


def test_core_weight_half_matches_hand_computed_blended_equity(tmp_path):
    # Arrange: 3 hourly bars, one round-trip trade (buy signal bar0, sell signal bar1), no fees,
    # execution on next bar's open (lag=1, the default).
    #
    # Execution trace:
    #   bar1 (j=bar0): buy fills at open[1]=100 -> units = 1/100 = 0.01, cash = 0.
    #                  strategy_equity[1] = units * close[1] = 0.01 * 100 = 1.0
    #   bar2 (j=bar1): sell fills at open[2]=102 -> cash = 0.01 * 102 = 1.02, units = 0.
    #                  strategy_equity[2] = cash = 1.02
    # Strategy equity: [1.0, 1.0, 1.02]. in_pos: [False, True, False] -> mean = 1/3.
    #
    # Core (buy&hold) sleeve from close = [100, 100, 104] -> core_equity = [1.0, 1.0, 1.04].
    #
    # Blended at w=0.5: [1.0, 1.0, 0.5*1.04 + 0.5*1.02] = [1.0, 1.0, 1.03] -> net_return_% = 3.0.
    # Exposure: 0.5 + 0.5 * (1/3) = 2/3 -> 66.7%.
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01T00:00:00Z", periods=3, freq="h").strftime(
            "%Y-%m-%dT%H:%M:%S.%f+00:00"
        ),
        "open": [100.0, 100.0, 102.0],
        "close": [100.0, 100.0, 104.0],
        "buy_signal_column": [True, False, False],
        "sell_signal_column": [False, True, False],
    })
    file = tmp_path / "signals.csv"
    df.to_csv(file, index=False)
    args = evaluate_signals.build_parser().parse_args(
        ["--file", str(file), "--core-weight", "0.5", "--fee", "0", "--slippage", "0"]
    )

    # Act
    result = evaluate_signals.run(args)["ALL"]

    # Assert
    assert result["net_return_%"] == pytest.approx(3.0)
    assert result["buy_hold_%"] == pytest.approx(4.0)  # unaffected by w: 104/100 - 1 = 4.0%
    assert result["exposure_%"] == pytest.approx(66.7)
    assert result["trades"] == 1
    assert result["avg_trade_%"] == pytest.approx(2.0)  # strategy-only, unaffected by w: 1.02/1 - 1


def test_holdout_split_reports_two_independent_subperiods(tmp_path):
    # Arrange: 48 hourly bars; holdout from bar 12, split at bar 30
    times = pd.date_range("2025-01-01T00:00:00Z", periods=48, freq="h")
    df = pd.DataFrame({
        "timestamp": times.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),
        "open": [100.0 + i for i in range(48)],
        "close": [100.5 + i for i in range(48)],
        "buy_signal_column": [i % 6 == 0 for i in range(48)],
        "sell_signal_column": [i % 6 == 3 for i in range(48)],
    })
    file = tmp_path / "signals.csv"
    df.to_csv(file, index=False)
    args = evaluate_signals.build_parser().parse_args([
        "--file", str(file), "--holdout-start", str(times[12]), "--holdout-split", str(times[30])])

    # Act
    result = evaluate_signals.run(args)

    # Assert
    assert result["HOLDOUT_A"]["bars"] == 18
    assert result["HOLDOUT_B"]["bars"] == 18
    assert result["HOLDOUT"]["bars"] == 36
    assert result["HOLDOUT_A"]["period"].startswith(str(times[12])[:19])
