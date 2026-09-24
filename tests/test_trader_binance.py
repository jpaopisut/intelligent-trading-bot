"""
Tests for outputs/trader_binance.py::new_limit_order

Covers a bug (B01) where the "no_trades_only_data_processing" (dry-run)
branch never assigned the local variable `order`, causing an
UnboundLocalError on `App.order = order` / `return order`.
"""
import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from binance.enums import SIDE_SELL

from service.App import App
from common.types import AccountBalances
from outputs.trader_binance import new_limit_order


def _make_fake_analyzer(close_price="100.0"):
    """Build a minimal fake analyzer exposing get_last_kline(symbol)."""
    # Kline is a list-like where index 4 is the close price (per production code).
    fake_kline = [None, None, None, None, close_price]
    analyzer = MagicMock()
    analyzer.get_last_kline.return_value = fake_kline
    return analyzer


def _setup_app(no_trades_only_data_processing: bool):
    App.config["symbol"] = "BTCUSDT"
    App.config["trade_model"] = {
        "no_trades_only_data_processing": no_trades_only_data_processing,
        "simulate_order_execution": True,  # avoid any real Binance client call
        "limit_price_adjustment": 0.001,
        "percentage_used_for_trade": 99,
    }
    App.analyzer = _make_fake_analyzer()
    App.account_info = AccountBalances()
    App.account_info.base_quantity = "0.01"
    App.order = "not-none-sentinel"


def test_new_limit_order_dry_run_returns_none_and_sets_app_order_none():
    """
    Arrange: dry-run mode (no_trades_only_data_processing=True), Binance client mocked out.
    Act: call new_limit_order(SIDE_SELL).
    Assert: returns None (no UnboundLocalError) and App.order is None.
    """
    _setup_app(no_trades_only_data_processing=True)

    with patch("outputs.trader_binance.collector_binance.client", MagicMock()):
        result = asyncio.run(new_limit_order(SIDE_SELL))

    assert result is None
    assert App.order is None


def test_new_limit_order_execute_path_still_returns_order():
    """
    Arrange: normal (non dry-run) mode with simulated order execution so no
    real network/order calls are made.
    Act: call new_limit_order(SIDE_SELL).
    Assert: the execute_order() path is unaffected - it still returns the
    order dict and stores it on App.order.
    """
    _setup_app(no_trades_only_data_processing=False)

    with patch("outputs.trader_binance.collector_binance.client", MagicMock()):
        result = asyncio.run(new_limit_order(SIDE_SELL))

    assert result is not None
    assert result["symbol"] == "BTCUSDT"
    assert result["side"] == SIDE_SELL
    assert App.order == result
