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
import sys
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
from outputs.trader_mt5 import trader_mt5


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
