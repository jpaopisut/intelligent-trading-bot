"""
Tests for outputs/trader_mt5.py::trader_mt5.

Covers a bug (B05, mirrors B02 fixed in trader_binance.py) where
"no_trades_only_data_processing" (dry-run) was read from two different
dicts: `model` (a generator's own config block, in trader_mt5()) vs.
`config["trade_model"]` (in new_limit_order()). The fix makes trader_mt5()
read the flag once from `config["trade_model"]` (the same dict
new_limit_order() uses) and reuse that single value for both the BUY and
SELL branches.

The MetaTrader5 package is not installed in this environment, so it is
stubbed out via sys.modules before outputs.trader_mt5 is imported.
"""
import asyncio
import copy
import sys
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# MetaTrader5 is not installed (and must not be) in this environment.
# Stub it out before importing outputs.trader_mt5, which does
# `import MetaTrader5 as mt5` at module load time.
if "MetaTrader5" not in sys.modules:
    mt5_stub = MagicMock()
    mt5_stub.__author__ = "test-stub"
    mt5_stub.__version__ = "0.0.0"
    mt5_stub.ORDER_TYPE_BUY_LIMIT = "BUY_LIMIT"
    mt5_stub.ORDER_TYPE_SELL = "SELL"
    mt5_stub.ORDER_TYPE_SELL_LIMIT = "SELL_LIMIT"
    sys.modules["MetaTrader5"] = mt5_stub

# inputs/collector_mt5.py has an unrelated, pre-existing module-level bug
# (`datetime(..., tzinfo=timezone)` where `timezone` resolves to the
# `datetime.timezone` class, not an instance - a TypeError at import time,
# independent of whether MetaTrader5 is installed). It is out of scope for
# B05, so stub the whole module here rather than fix it as a side effect of
# this change. outputs.trader_mt5 only needs `connect_mt5` from it, and our
# tests intentionally avoid the connect_mt5() code path (no mt5 credentials
# configured), so a no-op stub is sufficient.
if "inputs.collector_mt5" not in sys.modules:
    collector_mt5_stub = MagicMock()
    collector_mt5_stub.connect_mt5 = MagicMock(return_value=True)
    sys.modules["inputs.collector_mt5"] = collector_mt5_stub

from service.App import App
from outputs.trader_mt5 import trader_mt5, new_limit_order
import outputs.trader_mt5 as trader_mt5_module


@pytest.fixture(autouse=True)
def _restore_app_and_modules_state():
    """
    Restore App class attributes and sys.modules entries touched by this test
    module after each test, so state does not leak between tests in this file
    or into other test files/modules.

    App is a plain class with class-level attributes used as globals, and the
    tests here (and the module import above) mutate `sys.modules` for
    MetaTrader5/inputs.collector_mt5 and various App.* fields. Snapshot before
    the test, restore after, regardless of whether the test passed or failed.
    """
    app_attrs_to_snapshot = [
        "config", "status", "order", "order_time", "account_info", "analyzer",
    ]
    original_app_state = {
        attr: copy.deepcopy(getattr(App, attr, None))
        if attr == "config" else getattr(App, attr, None)
        for attr in app_attrs_to_snapshot
    }
    original_sys_modules = {
        name: sys.modules.get(name)
        for name in ("MetaTrader5", "inputs.collector_mt5")
    }

    try:
        yield
    finally:
        for attr, value in original_app_state.items():
            setattr(App, attr, value)
        for name, module in original_sys_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _make_buy_signal_df():
    """A 1-row df whose only row triggers a BUY signal."""
    index = pd.date_range("2024-01-01", periods=1, freq="1min")
    return pd.DataFrame(
        {"close": [100.0], "buy_col": [True], "sell_col": [False]},
        index=index,
    )


def _make_sell_signal_df():
    """A 1-row df whose only row triggers a SELL signal."""
    index = pd.date_range("2024-01-01", periods=1, freq="1min")
    return pd.DataFrame(
        {"close": [100.0], "buy_col": [False], "sell_col": [True]},
        index=index,
    )


def _setup_app_for_trader_mt5(no_trades_only_data_processing: bool, status: str):
    """
    Arrange full App/config state needed by trader_mt5(), with the dry-run
    flag set ONLY in config["trade_model"] (the canonical, single source of
    truth per B02/B05). If trader_mt5() were still (incorrectly) reading
    the flag from `model` (the generator config block), these tests would
    fail because `model` never sets this key.

    mt5_account_id/mt5_password/mt5_server are intentionally left unset so
    that trader_mt5() skips the connect_mt5() call entirely.
    """
    App.config["symbol"] = "EURUSD"
    App.config["freq"] = "1min"
    App.config["mt5_account_id"] = None
    App.config["mt5_password"] = None
    App.config["mt5_server"] = None
    App.config["trade_model"] = {
        "no_trades_only_data_processing": no_trades_only_data_processing,
        "limit_price_adjustment": 0.001,
        "percentage_used_for_trade": 99,
    }
    App.status = status
    App.order = None


# `model` intentionally does NOT contain "no_trades_only_data_processing":
# it only carries the generator's own signal-column config, mirroring how
# signal_sets entries are configured in practice.
_MODEL = {"buy_signal_column": "buy_col", "sell_signal_column": "sell_col"}


def test_trader_mt5_dry_run_flag_skips_status_change_on_buy():
    """
    Arrange: dry-run mode set only in config["trade_model"], BUY signal,
    status SOLD.
    Act: run trader_mt5().
    Assert: App.status is NOT advanced to "BUYING" because
    no_trades_only_data_processing must be read from config["trade_model"],
    not from `model` (which never carries this key).
    """
    _setup_app_for_trader_mt5(no_trades_only_data_processing=True, status="SOLD")
    df = _make_buy_signal_df()

    with patch("outputs.trader_mt5.update_account_balance", new=AsyncMock()), \
         patch("outputs.trader_mt5.new_limit_order", new=AsyncMock()) as mock_order:
        asyncio.run(trader_mt5(df, _MODEL, App.config, model_store=None))

    mock_order.assert_awaited_once()
    assert App.status == "SOLD"  # unchanged - status must not advance while skipping


def test_trader_mt5_dry_run_flag_skips_status_change_on_sell():
    """
    Arrange: dry-run mode set only in config["trade_model"], SELL signal,
    status BOUGHT.
    Act: run trader_mt5().
    Assert: App.status is NOT advanced to "SELLING".
    """
    _setup_app_for_trader_mt5(no_trades_only_data_processing=True, status="BOUGHT")
    df = _make_sell_signal_df()

    with patch("outputs.trader_mt5.update_account_balance", new=AsyncMock()), \
         patch("outputs.trader_mt5.new_limit_order", new=AsyncMock()) as mock_order:
        asyncio.run(trader_mt5(df, _MODEL, App.config, model_store=None))

    mock_order.assert_awaited_once()
    assert App.status == "BOUGHT"  # unchanged - status must not advance while skipping


def test_trader_mt5_flag_false_does_not_skip_status_change_on_buy():
    """
    Arrange: no_trades_only_data_processing=False, BUY signal, status SOLD.
    Act: run trader_mt5().
    Assert: App.status advances to "BUYING", i.e. trading is not skipped.
    """
    _setup_app_for_trader_mt5(no_trades_only_data_processing=False, status="SOLD")
    df = _make_buy_signal_df()

    with patch("outputs.trader_mt5.update_account_balance", new=AsyncMock()), \
         patch("outputs.trader_mt5.new_limit_order", new=AsyncMock()) as mock_order:
        asyncio.run(trader_mt5(df, _MODEL, App.config, model_store=None))

    mock_order.assert_awaited_once()
    assert App.status == "BUYING"


def test_trader_mt5_flag_false_does_not_skip_status_change_on_sell():
    """
    Arrange: no_trades_only_data_processing=False, SELL signal, status BOUGHT.
    Act: run trader_mt5().
    Assert: App.status advances to "SELLING".
    """
    _setup_app_for_trader_mt5(no_trades_only_data_processing=False, status="BOUGHT")
    df = _make_sell_signal_df()

    with patch("outputs.trader_mt5.update_account_balance", new=AsyncMock()), \
         patch("outputs.trader_mt5.new_limit_order", new=AsyncMock()) as mock_order:
        asyncio.run(trader_mt5(df, _MODEL, App.config, model_store=None))

    mock_order.assert_awaited_once()
    assert App.status == "SELLING"


#
# B11 - bug fixes in new_limit_order(): missing `symbol` arg on the BUY call,
# wrong order type (market SELL instead of SELL_LIMIT) on the SELL call, and
# an UnboundLocalError on the dry-run path (`order` never assigned before
# `App.order = order`).
#
# These tests exercise the REAL new_limit_order() code path (only the
# MetaTrader5 SDK boundary is mocked), so they would fail against the
# pre-fix code: bug 1 raises TypeError (missing positional `symbol`), bug 2
# would pass mt5.ORDER_TYPE_SELL instead of mt5.ORDER_TYPE_SELL_LIMIT in the
# submitted order spec, and bug 3 raises UnboundLocalError in dry-run mode.
#

def _setup_app_for_new_limit_order(no_trades_only_data_processing: bool):
    """Arrange full App state needed directly by new_limit_order()."""
    App.config["symbol"] = "EURUSD"
    App.config["trade_model"] = {
        "no_trades_only_data_processing": no_trades_only_data_processing,
        "limit_price_adjustment": 0.001,
        "percentage_used_for_trade": 99,
    }
    App.account_info = SimpleNamespace(
        quote_quantity=Decimal("1000"),
        base_quantity=Decimal("0.5"),
    )
    App.analyzer = MagicMock()
    App.analyzer.get_last_kline = MagicMock(return_value=[0, 0, 0, 0, "1.1000"])
    App.order = None
    App.order_time = None


def test_new_limit_order_buy_passes_symbol_and_buy_limit_type():
    """
    Arrange: non-dry-run BUY order.
    Act: call the real new_limit_order() with side=ORDER_TYPE_BUY_LIMIT.
    Assert: execute_order() is called with the correct symbol and order type
    (this is the call trader_mt5() makes for a BUY signal; bug 1 was that
    trader_mt5() omitted the required `symbol` positional argument here).
    """
    _setup_app_for_new_limit_order(no_trades_only_data_processing=False)

    with patch("outputs.trader_mt5.execute_order") as mock_execute_order:
        mock_execute_order.return_value = {"orderId": 1, "status": "NEW"}
        order = asyncio.run(
            new_limit_order(App.config["symbol"], side=trader_mt5_module.mt5.ORDER_TYPE_BUY_LIMIT)
        )

    mock_execute_order.assert_called_once()
    submitted_spec = mock_execute_order.call_args[0][0]
    assert submitted_spec["symbol"] == "EURUSD"
    assert submitted_spec["type"] == trader_mt5_module.mt5.ORDER_TYPE_BUY_LIMIT
    assert order == {"orderId": 1, "status": "NEW"}
    assert App.order == order


def test_new_limit_order_sell_uses_sell_limit_not_market_sell():
    """
    Arrange: non-dry-run SELL order.
    Act: call the real new_limit_order() with side=ORDER_TYPE_SELL_LIMIT, the
    value trader_mt5() must pass for its SELL branch (bug 2 was that it
    passed the market mt5.ORDER_TYPE_SELL instead, which new_limit_order's
    own price/quantity branching - written only for *_LIMIT types - does not
    handle).
    Assert: execute_order() receives type == ORDER_TYPE_SELL_LIMIT, and the
    quantity branch for SELL_LIMIT is used (all available base quantity).
    """
    _setup_app_for_new_limit_order(no_trades_only_data_processing=False)

    with patch("outputs.trader_mt5.execute_order") as mock_execute_order:
        mock_execute_order.return_value = {"orderId": 2, "status": "NEW"}
        order = asyncio.run(
            new_limit_order(App.config["symbol"], side=trader_mt5_module.mt5.ORDER_TYPE_SELL_LIMIT)
        )

    mock_execute_order.assert_called_once()
    submitted_spec = mock_execute_order.call_args[0][0]
    assert submitted_spec["symbol"] == "EURUSD"
    assert submitted_spec["type"] == trader_mt5_module.mt5.ORDER_TYPE_SELL_LIMIT
    assert submitted_spec["type"] != trader_mt5_module.mt5.ORDER_TYPE_SELL
    assert submitted_spec["volume"] == float(App.account_info.base_quantity)
    assert order == {"orderId": 2, "status": "NEW"}


def test_new_limit_order_dry_run_does_not_crash_and_sets_app_order():
    """
    Arrange: dry-run mode (no_trades_only_data_processing=True), BUY order.
    Act: call the real new_limit_order().
    Assert: it does not raise (bug 3 was UnboundLocalError: `order` was
    referenced in `App.order = order` without ever being assigned on the
    dry-run branch), App.order is set to the order spec that would have been
    submitted, and execute_order() (the real submission path) is never
    called.
    """
    _setup_app_for_new_limit_order(no_trades_only_data_processing=True)

    with patch("outputs.trader_mt5.execute_order") as mock_execute_order:
        order = asyncio.run(
            new_limit_order(App.config["symbol"], side=trader_mt5_module.mt5.ORDER_TYPE_BUY_LIMIT)
        )

    mock_execute_order.assert_not_called()
    assert order is not None
    assert order["symbol"] == "EURUSD"
    assert order["type"] == trader_mt5_module.mt5.ORDER_TYPE_BUY_LIMIT
    assert App.order == order


def test_trader_mt5_end_to_end_buy_calls_new_limit_order_without_typeerror():
    """
    Arrange: full trader_mt5() flow, real new_limit_order(), BUY signal,
    non-dry-run.
    Act: run trader_mt5().
    Assert: no TypeError is raised (bug 1 - missing `symbol` positional
    argument on the BUY call site) and the order actually submitted (via the
    mocked execute_order) carries the BUY_LIMIT type and correct symbol.
    """
    _setup_app_for_trader_mt5(no_trades_only_data_processing=False, status="SOLD")
    App.account_info = SimpleNamespace(
        quote_quantity=Decimal("1000"),
        base_quantity=Decimal("0.5"),
    )
    App.analyzer = MagicMock()
    App.analyzer.get_last_kline = MagicMock(return_value=[0, 0, 0, 0, "1.1000"])
    df = _make_buy_signal_df()

    with patch("outputs.trader_mt5.update_account_balance", new=AsyncMock()), \
         patch("outputs.trader_mt5.execute_order") as mock_execute_order:
        mock_execute_order.return_value = {"orderId": 3, "status": "NEW"}
        asyncio.run(trader_mt5(df, _MODEL, App.config, model_store=None))

    mock_execute_order.assert_called_once()
    submitted_spec = mock_execute_order.call_args[0][0]
    assert submitted_spec["symbol"] == "EURUSD"
    assert submitted_spec["type"] == trader_mt5_module.mt5.ORDER_TYPE_BUY_LIMIT
    assert App.status == "BUYING"


def test_trader_mt5_end_to_end_sell_uses_sell_limit_type():
    """
    Arrange: full trader_mt5() flow, real new_limit_order(), SELL signal,
    non-dry-run.
    Act: run trader_mt5().
    Assert: the order submitted uses ORDER_TYPE_SELL_LIMIT, not the market
    ORDER_TYPE_SELL (bug 2).
    """
    _setup_app_for_trader_mt5(no_trades_only_data_processing=False, status="BOUGHT")
    App.account_info = SimpleNamespace(
        quote_quantity=Decimal("1000"),
        base_quantity=Decimal("0.5"),
    )
    App.analyzer = MagicMock()
    App.analyzer.get_last_kline = MagicMock(return_value=[0, 0, 0, 0, "1.1000"])
    df = _make_sell_signal_df()

    with patch("outputs.trader_mt5.update_account_balance", new=AsyncMock()), \
         patch("outputs.trader_mt5.execute_order") as mock_execute_order:
        mock_execute_order.return_value = {"orderId": 4, "status": "NEW"}
        asyncio.run(trader_mt5(df, _MODEL, App.config, model_store=None))

    mock_execute_order.assert_called_once()
    submitted_spec = mock_execute_order.call_args[0][0]
    assert submitted_spec["symbol"] == "EURUSD"
    assert submitted_spec["type"] == trader_mt5_module.mt5.ORDER_TYPE_SELL_LIMIT
    assert submitted_spec["type"] != trader_mt5_module.mt5.ORDER_TYPE_SELL
    assert App.status == "SELLING"
