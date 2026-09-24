"""
Tests for outputs/trader_binance.py::new_limit_order and trader_binance.

Covers a bug (B01) where the "no_trades_only_data_processing" (dry-run)
branch never assigned the local variable `order`, causing an
UnboundLocalError on `App.order = order` / `return order`.

Also covers a bug (B02) where "no_trades_only_data_processing" was read
from two different dicts: `model` (a generator's own config block, in
trader_binance()) vs. `config["trade_model"]` (in new_limit_order()). The
fix makes trader_binance() read the flag once from `config["trade_model"]`
(the same dict new_limit_order() uses) and reuse that single value for
both the BUY and SELL branches.
"""
import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from binance.enums import SIDE_BUY, SIDE_SELL
from binance.exceptions import BinanceAPIException

from service.App import App
from common.types import AccountBalances
from outputs.trader_binance import _generate_client_order_id, new_limit_order, trader_binance


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

    with patch("outputs.trader_binance.collector_binance.client", _make_fake_binance_client()):
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

    with patch("outputs.trader_binance.collector_binance.client", _make_fake_binance_client()):
        result = asyncio.run(new_limit_order(SIDE_SELL))

    assert result is not None
    assert result["symbol"] == "BTCUSDT"
    assert result["side"] == SIDE_SELL
    assert App.order == result


def _make_buy_signal_df():
    """A 1-row df whose only row triggers a BUY signal."""
    index = pd.date_range("2024-01-01", periods=1, freq="1min")
    return pd.DataFrame(
        {"close": [100.0], "buy_col": [True], "sell_col": [False]},
        index=index,
    )


def _setup_app_for_trader_binance(no_trades_only_data_processing: bool):
    """
    Arrange full App/config state needed by trader_binance(), with the
    dry-run flag set ONLY in config["trade_model"] (the canonical, single
    source of truth per B02). If trader_binance() were still (incorrectly)
    reading the flag from `model` (the generator config block), these
    tests would fail because `model` never sets this key.
    """
    App.config["symbol"] = "BTCUSDT"
    App.config["freq"] = "1min"
    App.config["base_asset"] = "BTC"
    App.config["quote_asset"] = "USDT"
    App.config["trade_model"] = {
        "no_trades_only_data_processing": no_trades_only_data_processing,
        "simulate_order_execution": True,  # avoid any real Binance client call
        "test_order_before_submit": False,
        "limit_price_adjustment": 0.001,
        "percentage_used_for_trade": 99,
    }
    App.analyzer = _make_fake_analyzer()
    App.account_info = AccountBalances()
    App.account_info.base_quantity = Decimal("0.01")
    App.account_info.quote_quantity = Decimal("1000")
    App.order = None
    App.status = "SOLD"  # So that a BUY signal triggers a new BUY order


def _make_fake_binance_client(min_notional="1.00000000"):
    client = MagicMock()
    client.get_asset_balance.side_effect = lambda asset: {"free": "1000.00000000"}
    client.get_symbol_info.return_value = _make_btcusdt_symbol_info(min_notional=min_notional)
    return client


def _make_btcusdt_symbol_info(min_notional_key="MIN_NOTIONAL", min_notional="1.00000000"):
    """
    A realistic (trimmed) Binance BTCUSDT-like exchangeInfo symbol entry.
    `min_notional_key` lets tests exercise both the legacy "MIN_NOTIONAL"
    filter name and the newer "NOTIONAL" filter name Binance uses on some
    symbols now.
    """
    return {
        "symbol": "BTCUSDT",
        "filters": [
            {
                "filterType": "PRICE_FILTER",
                "minPrice": "0.01000000",
                "maxPrice": "1000000.00000000",
                "tickSize": "0.01000000",
            },
            {
                "filterType": "LOT_SIZE",
                "minQty": "0.00001000",
                "maxQty": "9000.00000000",
                "stepSize": "0.00001000",
            },
            {
                "filterType": min_notional_key,
                "minNotional": min_notional,
                "applyToMarket": True,
            },
        ],
    }


# `model` intentionally does NOT contain "no_trades_only_data_processing":
# it only carries the generator's own signal-column config, mirroring how
# signal_sets entries are configured in practice.
_MODEL = {"buy_signal_column": "buy_col", "sell_signal_column": "sell_col"}


def test_trader_binance_dry_run_flag_skips_status_change_consistently():
    """
    Arrange: dry-run mode set only in config["trade_model"], BUY signal, status SOLD.
    Act: run trader_binance().
    Assert: new_limit_order() itself performs no real submission (App.order
    stays None, per new_limit_order's own dry-run branch) AND the caller in
    trader_binance() also treats it as skipped (App.status is NOT advanced
    to "BUYING"). Both reads agree because there is now a single source.
    """
    _setup_app_for_trader_binance(no_trades_only_data_processing=True)
    df = _make_buy_signal_df()

    with patch("outputs.trader_binance.collector_binance.client", _make_fake_binance_client()):
        asyncio.run(trader_binance(df, _MODEL, App.config, model_store=None))

    assert App.order is None
    assert App.status == "SOLD"  # unchanged - status must not advance while skipping


def test_trader_binance_flag_false_does_not_skip_order_submission():
    """
    Arrange: no_trades_only_data_processing=False, BUY signal, status SOLD.
    Act: run trader_binance().
    Assert: the order is actually created (App.order set, side=BUY) and the
    status advances to "BUYING", i.e. trading is not skipped.
    """
    _setup_app_for_trader_binance(no_trades_only_data_processing=False)
    df = _make_buy_signal_df()

    with patch("outputs.trader_binance.collector_binance.client", _make_fake_binance_client()):
        asyncio.run(trader_binance(df, _MODEL, App.config, model_store=None))

    assert App.order is not None
    assert App.order["side"] == SIDE_BUY
    assert App.status == "BUYING"


# ---------------------------------------------------------------------------
# B03: exchange filters (PRICE_FILTER.tickSize, LOT_SIZE.stepSize,
# MIN_NOTIONAL/NOTIONAL.minNotional) must be consulted, instead of the
# hardcoded round(2)/round_down(6) that ignored the exchange's real rules.
# ---------------------------------------------------------------------------

def _setup_app_for_filters(base_quantity, close_price="100.03"):
    App.config["symbol"] = "BTCUSDT"
    App.config["trade_model"] = {
        "no_trades_only_data_processing": False,
        "simulate_order_execution": True,  # avoid any real Binance client call
        "limit_price_adjustment": 0.001,
        "percentage_used_for_trade": 99,
    }
    App.analyzer = _make_fake_analyzer(close_price=close_price)
    App.account_info = AccountBalances()
    App.account_info.base_quantity = Decimal(base_quantity)
    App.account_info.quote_quantity = Decimal("1000")
    App.order = "not-none-sentinel"


def test_new_limit_order_rounds_price_and_quantity_to_exchange_tick_and_step():
    """
    Arrange: SELL order whose naive price/quantity do not land on tickSize
    (0.05) / stepSize (0.0001) boundaries.
    Act: call new_limit_order(SIDE_SELL).
    Assert: price and quantity are floored (rounded down) to the nearest
    tick/step using Decimal arithmetic, not the old hardcoded round(2)/
    round_down(6).
    """
    _setup_app_for_filters(base_quantity="0.123456789", close_price="100.03")
    # price = 100.03 * 1.001 = 100.13003 -> floored to nearest 0.05 = 100.10
    # quantity = 0.123456789 -> floored to nearest 0.0001 = 0.1234
    client = MagicMock()
    client.get_symbol_info.return_value = _make_btcusdt_symbol_info(min_notional="1.00000000")
    client.get_symbol_info.return_value["filters"][0]["tickSize"] = "0.05000000"
    client.get_symbol_info.return_value["filters"][1]["stepSize"] = "0.00010000"

    with patch("outputs.trader_binance.collector_binance.client", client):
        result = asyncio.run(new_limit_order(SIDE_SELL))

    assert result is not None
    assert result["price"] == "100.10"
    assert result["quantity"] == "0.1234"


def test_new_limit_order_below_min_notional_returns_none_and_does_not_submit():
    """
    Arrange: price * quantity computed below the exchange's minNotional.
    Act: call new_limit_order(SIDE_SELL).
    Assert: the order is not submitted (no create_order/create_test_order
    call), the function returns None and App.order is set to None.
    """
    _setup_app_for_filters(base_quantity="0.0001", close_price="100.0")
    # price ~= 100.1, quantity = 0.0001 -> notional ~= 0.01001, well below 10.
    client = MagicMock()
    client.get_symbol_info.return_value = _make_btcusdt_symbol_info(min_notional="10.00000000")

    with patch("outputs.trader_binance.collector_binance.client", client):
        result = asyncio.run(new_limit_order(SIDE_SELL))

    assert result is None
    assert App.order is None
    client.create_order.assert_not_called()
    client.create_test_order.assert_not_called()


def test_new_limit_order_at_min_notional_proceeds_normally():
    """
    Arrange: price * quantity computed at/above minNotional.
    Act: call new_limit_order(SIDE_SELL).
    Assert: existing (pre-B03) behavior is preserved - the order is built
    and returned.
    """
    _setup_app_for_filters(base_quantity="1.0", close_price="100.0")
    # price ~= 100.1, quantity = 1.0 -> notional ~= 100.1, well above 10.
    client = MagicMock()
    client.get_symbol_info.return_value = _make_btcusdt_symbol_info(min_notional="10.00000000")

    with patch("outputs.trader_binance.collector_binance.client", client):
        result = asyncio.run(new_limit_order(SIDE_SELL))

    assert result is not None
    assert result["side"] == SIDE_SELL
    assert App.order == result


# ---------------------------------------------------------------------------
# B04: order-state safety.
# 1) Callers in trader_binance() must not flip App.status to BUYING/SELLING
#    when new_limit_order() returns None (order not actually created).
# 2) Every early-return path inside new_limit_order() must clear App.order
#    (not leave a stale, previously-FILLED order dict visible).
# ---------------------------------------------------------------------------

def test_trader_binance_does_not_set_buying_status_when_order_rejected():
    """
    Arrange: BUY signal, status SOLD, but the order will be rejected because
    its notional is below the exchange's minNotional (new_limit_order
    returns None).
    Act: run trader_binance().
    Assert: App.status stays "SOLD" - it must NOT be advanced to "BUYING"
    just because trading was attempted; only a real returned order may
    advance status.
    """
    _setup_app_for_trader_binance(no_trades_only_data_processing=False)
    df = _make_buy_signal_df()

    # update_account_balance() always overwrites App.account_info with the
    # fixed "1000.00000000" balance from the mocked client, so force a
    # reject via an exchange minNotional above the resulting order notional
    # (~990 quote units) instead of via account balance.
    client = _make_fake_binance_client(min_notional="2000.00000000")

    with patch("outputs.trader_binance.collector_binance.client", client):
        asyncio.run(trader_binance(df, _MODEL, App.config, model_store=None))

    assert App.order is None
    assert App.status == "SOLD"


def test_trader_binance_sets_buying_status_when_order_created():
    """
    Arrange: BUY signal, status SOLD, order will actually be created
    (notional comfortably above minNotional).
    Act: run trader_binance().
    Assert: App.status advances to "BUYING" because new_limit_order()
    returned a real order.
    """
    _setup_app_for_trader_binance(no_trades_only_data_processing=False)
    df = _make_buy_signal_df()

    client = _make_fake_binance_client(min_notional="1.00000000")

    with patch("outputs.trader_binance.collector_binance.client", client):
        asyncio.run(trader_binance(df, _MODEL, App.config, model_store=None))

    assert App.order is not None
    assert App.status == "BUYING"


def _setup_app_with_stale_order():
    """Same as _setup_app_for_filters but pre-seeds App.order with a stale FILLED order."""
    _setup_app_for_filters(base_quantity="1.0", close_price="100.0")
    App.order = {"symbol": "BTCUSDT", "side": SIDE_SELL, "status": "FILLED", "orderId": 1}


def test_new_limit_order_clears_stale_order_when_symbol_info_missing():
    """
    Arrange: App.order pre-set to a stale FILLED order; client returns no
    symbol info at all.
    Act: call new_limit_order(SIDE_SELL).
    Assert: returns None and App.order is cleared to None (no stale order
    left visible).
    """
    _setup_app_with_stale_order()
    client = MagicMock()
    client.get_symbol_info.return_value = None

    with patch("outputs.trader_binance.collector_binance.client", client):
        result = asyncio.run(new_limit_order(SIDE_SELL))

    assert result is None
    assert App.order is None


def test_new_limit_order_clears_stale_order_when_filters_missing():
    """
    Arrange: App.order pre-set to a stale FILLED order; symbol info is
    missing the PRICE_FILTER/LOT_SIZE filters.
    Act: call new_limit_order(SIDE_SELL).
    Assert: returns None and App.order is cleared to None.
    """
    _setup_app_with_stale_order()
    client = MagicMock()
    client.get_symbol_info.return_value = {"symbol": "BTCUSDT", "filters": []}

    with patch("outputs.trader_binance.collector_binance.client", client):
        result = asyncio.run(new_limit_order(SIDE_SELL))

    assert result is None
    assert App.order is None


def test_new_limit_order_clears_stale_order_when_close_price_missing():
    """
    Arrange: App.order pre-set to a stale FILLED order; the analyzer's last
    kline has a close price of "0" (to_decimal("0") == Decimal(0), falsy).
    Act: call new_limit_order(SIDE_SELL).
    Assert: returns None and App.order is cleared to None.
    """
    _setup_app_with_stale_order()
    App.analyzer = _make_fake_analyzer(close_price="0")
    client = MagicMock()
    client.get_symbol_info.return_value = _make_btcusdt_symbol_info(min_notional="1.00000000")

    with patch("outputs.trader_binance.collector_binance.client", client):
        result = asyncio.run(new_limit_order(SIDE_SELL))

    assert result is None
    assert App.order is None


def test_new_limit_order_handles_notional_filter_renamed_to_notional():
    """
    Arrange: symbol info uses the newer "NOTIONAL" filter name (Binance
    renamed MIN_NOTIONAL on some symbols) instead of "MIN_NOTIONAL".
    Act: call new_limit_order(SIDE_SELL) with a notional below the limit.
    Assert: the "NOTIONAL" filter is still honored and the order is blocked.
    """
    _setup_app_for_filters(base_quantity="0.0001", close_price="100.0")
    client = MagicMock()
    client.get_symbol_info.return_value = _make_btcusdt_symbol_info(
        min_notional_key="NOTIONAL", min_notional="10.00000000"
    )

    with patch("outputs.trader_binance.collector_binance.client", client):
        result = asyncio.run(new_limit_order(SIDE_SELL))

    assert result is None
    assert App.order is None


# ---------------------------------------------------------------------------
# B09: create_order() must be idempotent via a deterministic newClientOrderId,
# and any create_order() failure/timeout must be reconciled with the exchange
# (via get_order(origClientOrderId=...)) instead of assumed to have failed.
# ---------------------------------------------------------------------------

def _make_binance_api_exception(code: int, message: str = "error"):
    """Build a BinanceAPIException carrying a given Binance error `code`."""
    text = f'{{"code": {code}, "msg": "{message}"}}'
    response = MagicMock()
    response.text = text
    return BinanceAPIException(response, 400, text)


def test_generate_client_order_id_is_deterministic_and_within_length_limit():
    """
    Same (symbol, side, close_time) must always produce the same id (needed for
    idempotent retries), and the id must respect Binance's 36-char clientOrderId limit.
    """
    id1 = _generate_client_order_id("BTCUSDT", SIDE_BUY, "2024-01-01T00:01:00")
    id2 = _generate_client_order_id("BTCUSDT", SIDE_BUY, "2024-01-01T00:01:00")
    id3 = _generate_client_order_id("BTCUSDT", SIDE_SELL, "2024-01-01T00:01:00")

    assert id1 == id2
    assert id1 != id3
    assert len(id1) <= 36


def test_new_limit_order_success_path_passes_deterministic_client_order_id():
    """
    Regression test: on a normal successful create_order() call, the order is
    submitted with a newClientOrderId derived from (symbol, side, close_time),
    and App.order reflects the returned order (no reconciliation needed).
    """
    _setup_app_for_filters(base_quantity="1.0", close_price="100.0")
    App.config["trade_model"]["simulate_order_execution"] = False
    close_time = "2024-01-01T00:05:00"
    expected_id = _generate_client_order_id("BTCUSDT", SIDE_SELL, close_time)

    client = MagicMock()
    client.get_symbol_info.return_value = _make_btcusdt_symbol_info(min_notional="1.00000000")
    returned_order = {
        "symbol": "BTCUSDT", "side": SIDE_SELL, "status": "NEW", "orderId": 42,
        "newClientOrderId": expected_id,
    }
    client.create_order.return_value = returned_order

    with patch("outputs.trader_binance.collector_binance.client", client):
        result = asyncio.run(new_limit_order(SIDE_SELL, close_time=close_time))

    assert result == returned_order
    assert App.order == returned_order
    _, kwargs = client.create_order.call_args
    assert kwargs["newClientOrderId"] == expected_id
    client.get_order.assert_not_called()


def test_new_limit_order_create_order_exception_then_order_found_is_adopted():
    """
    Arrange: create_order() raises (e.g. a timeout - the outcome on the exchange
    is unknown).
    Act: new_limit_order() must NOT assume the order failed; it looks up the order
    by the same clientOrderId and, since it exists on the exchange, adopts it
    instead of retrying create_order().
    Assert: the looked-up order is returned/stored, create_order() was called
    exactly once (no blind retry/duplicate), and get_order() used the same id.
    """
    _setup_app_for_filters(base_quantity="1.0", close_price="100.0")
    App.config["trade_model"]["simulate_order_execution"] = False
    close_time = "2024-01-01T00:06:00"
    expected_id = _generate_client_order_id("BTCUSDT", SIDE_SELL, close_time)

    client = MagicMock()
    client.get_symbol_info.return_value = _make_btcusdt_symbol_info(min_notional="1.00000000")
    client.create_order.side_effect = TimeoutError("request timed out")
    existing_order = {
        "symbol": "BTCUSDT", "side": SIDE_SELL, "status": "NEW", "orderId": 99,
        "newClientOrderId": expected_id,
    }
    client.get_order.return_value = existing_order

    with patch("outputs.trader_binance.collector_binance.client", client):
        result = asyncio.run(new_limit_order(SIDE_SELL, close_time=close_time))

    assert result == existing_order
    assert App.order == existing_order
    client.get_order.assert_called_once_with(symbol="BTCUSDT", origClientOrderId=expected_id)
    client.create_order.assert_called_once()


def test_new_limit_order_create_order_exception_then_order_not_found_leaves_no_order_state():
    """
    Arrange: create_order() raises (timeout), and the follow-up get_order() lookup
    reports the order does not exist on the exchange (Binance error code -2013).
    Act/Assert: new_limit_order() falls back to the established B04 "no order"
    state (returns None, App.order cleared) - no exception propagates.
    """
    _setup_app_for_filters(base_quantity="1.0", close_price="100.0")
    App.config["trade_model"]["simulate_order_execution"] = False
    close_time = "2024-01-01T00:07:00"
    expected_id = _generate_client_order_id("BTCUSDT", SIDE_SELL, close_time)

    client = MagicMock()
    client.get_symbol_info.return_value = _make_btcusdt_symbol_info(min_notional="1.00000000")
    client.create_order.side_effect = TimeoutError("request timed out")
    client.get_order.side_effect = _make_binance_api_exception(-2013, "Order does not exist.")

    with patch("outputs.trader_binance.collector_binance.client", client):
        result = asyncio.run(new_limit_order(SIDE_SELL, close_time=close_time))

    assert result is None
    assert App.order is None
    client.get_order.assert_called_once_with(symbol="BTCUSDT", origClientOrderId=expected_id)


def test_new_limit_order_duplicate_client_order_id_adopts_existing_order():
    """
    Arrange: create_order() raises a duplicate newClientOrderId error (Binance
    error code -2010), e.g. because a previous submission with the same
    deterministic id already went through (a retried cycle after a restart).
    Act: new_limit_order() must fetch and adopt the existing order via
    get_order() instead of creating a duplicate order.
    Assert: the existing order is returned/stored and create_order() was called
    exactly once (no duplicate submission).
    """
    _setup_app_for_filters(base_quantity="1.0", close_price="100.0")
    App.config["trade_model"]["simulate_order_execution"] = False
    close_time = "2024-01-01T00:08:00"
    expected_id = _generate_client_order_id("BTCUSDT", SIDE_SELL, close_time)

    client = MagicMock()
    client.get_symbol_info.return_value = _make_btcusdt_symbol_info(min_notional="1.00000000")
    client.create_order.side_effect = _make_binance_api_exception(-2010, "Duplicate order sent.")
    existing_order = {
        "symbol": "BTCUSDT", "side": SIDE_SELL, "status": "NEW", "orderId": 123,
        "newClientOrderId": expected_id,
    }
    client.get_order.return_value = existing_order

    with patch("outputs.trader_binance.collector_binance.client", client):
        result = asyncio.run(new_limit_order(SIDE_SELL, close_time=close_time))

    assert result == existing_order
    assert App.order == existing_order
    client.get_order.assert_called_once_with(symbol="BTCUSDT", origClientOrderId=expected_id)
    client.create_order.assert_called_once()
