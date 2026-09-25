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


# --- G06: idle-gated, time-varying core weight -----------------------------------------------

def _idle_gated_fixture(tmp_path: Path) -> Path:
    # 6 hourly bars, one round-trip trade (buy signal bar0, sell signal bar1), no fees,
    # execution on next bar's open (lag=1, the default). Same execution trace shape as the
    # G02 hand-computed test, extended with 3 extra flat bars after the exit to exercise idle-gating.
    #
    # Execution trace:
    #   bar1 (j=bar0): buy fills at open[1]=100 -> units=0.01, cash=0. strategy_equity[1]=1.0
    #   bar2 (j=bar1): sell fills at open[2]=102 -> cash=0.01*102=1.02, units=0. strategy_equity[2]=1.02
    # Strategy equity: [1.0, 1.0, 1.02, 1.02, 1.02, 1.02]. in_pos: [F, T, F, F, F, F].
    # idle (consecutive flat bars, execution-timed, causal): [1, 0, 1, 2, 3, 4].
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01T00:00:00Z", periods=6, freq="h").strftime(
            "%Y-%m-%dT%H:%M:%S.%f+00:00"
        ),
        "open":  [100.0, 100.0, 102.0, 103.0, 104.0, 105.0],
        "close": [100.0, 100.0, 104.0, 104.0, 106.0, 107.0],
        "buy_signal_column":  [True, False, False, False, False, False],
        "sell_signal_column": [False, True, False, False, False, False],
    })
    file = tmp_path / "signals.csv"
    df.to_csv(file, index=False)
    return file


def test_core_idle_bars_unset_or_zero_reproduces_exact_g02_constant_blend(tmp_path):
    # Arrange: the exact G02 fixture and hand-computed numbers from
    # test_core_weight_half_matches_hand_computed_blended_equity (net_return_%=3.0, buy_hold_%=4.0,
    # exposure_%=66.7).
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
    no_flag_args = evaluate_signals.build_parser().parse_args(
        ["--file", str(file), "--core-weight", "0.5", "--fee", "0", "--slippage", "0"]
    )
    unset_args = evaluate_signals.build_parser().parse_args(
        ["--file", str(file), "--core-weight", "0.5", "--fee", "0", "--slippage", "0"]
    )
    zero_args = evaluate_signals.build_parser().parse_args(
        ["--file", str(file), "--core-weight", "0.5", "--core-idle-bars", "0", "--fee", "0", "--slippage", "0"]
    )

    # Act
    no_flag = evaluate_signals.run(no_flag_args)["ALL"]
    unset = evaluate_signals.run(unset_args)["ALL"]
    zero = evaluate_signals.run(zero_args)["ALL"]

    # Assert: byte-for-byte identical to each other and to the G02 hand-computed numbers.
    assert no_flag == unset == zero
    assert zero["net_return_%"] == pytest.approx(3.0)
    assert zero["buy_hold_%"] == pytest.approx(4.0)
    assert zero["exposure_%"] == pytest.approx(66.7)
    # G02 semantics: core sleeve is blended on every bar, not gated -> reported as always active.
    assert zero["core_active_bars_%"] == 100.0
    assert zero["core_switches"] == 0
    assert zero["core_switch_fees_%"] == 0.0


def test_core_idle_bars_hand_computed_switches_and_switch_fees(tmp_path):
    # Arrange: see _idle_gated_fixture for the execution trace and idle counts.
    # N=2, core_weight=0.4 -> w_t qualifies (0.4) only once idle_t>=2, i.e. bars 3,4,5 (0-indexed).
    # w = [0.0, 0.0, 0.0, 0.4, 0.4, 0.4] (bar1 is in-position -> forced 0 regardless of idle).
    # core_ret (from close) = [0, 0, 0.04, 0, 0.019230769..., 0.009433962...]
    # strat_ret (from strategy equity) = [0, 0, 0.02, 0, 0, 0]
    # r_t = w_t*core_ret_t + (1-w_t)*strat_ret_t = [0, 0, 0.02, 0, 0.0076923..., 0.0037735...]
    # delta_w = [0, 0, 0, 0.4, 0, 0] -> one switch (turning on), switch_fee = 0.0005*0.4 = 0.0002 at bar3.
    # Compounding gives blended equity [1.0, 1.0, 1.02, 1.0197960..., 1.0276405846..., 1.0315184736...]
    # -> net_return_% = 3.15, fee_drag = 0.000204 -> core_switch_fees_% = 0.02.
    file = _idle_gated_fixture(tmp_path)
    args = evaluate_signals.build_parser().parse_args(
        ["--file", str(file), "--core-weight", "0.4", "--core-idle-bars", "2", "--fee", "0", "--slippage", "0"]
    )

    # Act
    result = evaluate_signals.run(args)["ALL"]

    # Assert
    assert result["net_return_%"] == pytest.approx(3.15)
    assert result["core_active_bars_%"] == pytest.approx(50.0)
    assert result["core_switches"] == 1
    assert result["core_switch_fees_%"] == pytest.approx(0.02)
    assert result["exposure_%"] == pytest.approx(36.7)
    # Trade stats stay strategy-only, unaffected by the core sleeve (same as G02).
    assert result["trades"] == 1
    assert result["avg_trade_%"] == pytest.approx(2.0)
    assert result["win_rate_%"] == pytest.approx(100.0)


def test_core_idle_bars_no_look_ahead(tmp_path):
    # Arrange: the same idle-gated fixture, plus a variant where a *future* bar (index 5, the last
    # one) has a wildly different price and signal. Only bars 0..4 are compared.
    file_a = _idle_gated_fixture(tmp_path)
    df_b = pd.read_csv(file_a)
    df_b.loc[5, ["open", "close"]] = [9999.0, 5.0]
    df_b.loc[5, "buy_signal_column"] = True
    file_b = tmp_path / "signals_shifted.csv"
    df_b.to_csv(file_b, index=False)

    common_kwargs = ["--core-weight", "0.4", "--core-idle-bars", "2", "--fee", "0", "--slippage", "0"]
    args_a = evaluate_signals.build_parser().parse_args(["--file", str(file_a)] + common_kwargs)
    args_b = evaluate_signals.build_parser().parse_args(["--file", str(file_b)] + common_kwargs)

    df_a = evaluate_signals.load_frame(file_a, "timestamp")
    df_b_loaded = evaluate_signals.load_frame(file_b, "timestamp")
    sim_a = evaluate_signals.simulate(df_a, args_a)
    sim_b = evaluate_signals.simulate(df_b_loaded, args_b)
    idle_a = evaluate_signals.compute_idle(sim_a["in_pos"])
    idle_b = evaluate_signals.compute_idle(sim_b["in_pos"])
    blend_a = evaluate_signals.core_weight_blend(df_a, sim_a, args_a)
    blend_b = evaluate_signals.core_weight_blend(df_b_loaded, sim_b, args_b)

    # Act / Assert: idle_t, w_t and the blended equity for bars 0..4 are unaffected by bar 5.
    assert list(idle_a[:5]) == list(idle_b[:5])
    assert list(blend_a["w"][:5]) == list(blend_b["w"][:5])
    assert blend_a["equity"][:5] == pytest.approx(blend_b["equity"][:5])


def test_core_weight_zero_is_bit_identical_regardless_of_core_idle_bars(tmp_path):
    # Arrange
    file = _idle_gated_fixture(tmp_path)
    no_idle_args = evaluate_signals.build_parser().parse_args(
        ["--file", str(file), "--core-weight", "0", "--fee", "0", "--slippage", "0"]
    )
    with_idle_args = evaluate_signals.build_parser().parse_args(
        ["--file", str(file), "--core-weight", "0", "--core-idle-bars", "3", "--fee", "0", "--slippage", "0"]
    )

    # Act
    baseline = evaluate_signals.run(no_idle_args)["ALL"]
    with_idle = evaluate_signals.run(with_idle_args)["ALL"]

    # Assert: bit-identical (w=0 short-circuit bypasses the idle-gating machinery entirely)
    assert with_idle == baseline
    assert with_idle["core_active_bars_%"] == 0.0
    assert with_idle["core_switches"] == 0
    assert with_idle["core_switch_fees_%"] == 0.0
