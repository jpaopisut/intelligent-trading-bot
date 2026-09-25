import os
import sys
import argparse
import hashlib
import math, time
from datetime import datetime
from decimal import *
from typing import Union

import pandas as pd
import asyncio

from binance import Client
from binance.exceptions import *
from binance.helpers import date_to_milliseconds, interval_to_milliseconds
from binance.enums import *

from service.App import *
from common.utils import *
from common.model_store import *
from inputs import collector_binance
from outputs.notifier_trades import get_signal

import logging
log = logging.getLogger('trader')


async def trader_binance(df, model: dict, config: dict, model_store: ModelStore):
    """
    It is a highest level task which is added to the event loop and executed normally every 1 minute and then it calls other tasks.
    """
    symbol = config["symbol"]
    freq = config["freq"]
    startTime, endTime = pandas_get_interval(freq)
    now_ts = now_timestamp()

    buy_signal_column = model.get("buy_signal_column")
    sell_signal_column = model.get("sell_signal_column")

    # Single canonical source for this flag: config["trade_model"], the same
    # dict read by new_limit_order()/execute_order(). Do NOT read it from
    # `model` (the generator's own config block) - that is a different dict
    # and was the cause of inconsistent dry-run behavior (B02).
    no_trades_only_data_processing = config.get("trade_model", {}).get("no_trades_only_data_processing")

    signal = get_signal(df, buy_signal_column, sell_signal_column)
    signal_side = signal.get("side")
    close_price = signal.get("close_price")
    close_time = signal.get("close_time")

    log.info(f"===> Start trade task. Timestamp {now_ts}. Interval [{startTime},{endTime}].")

    #
    # Sync trade status, check running orders (orders, account etc.)
    #
    status = App.status

    if status == "BUYING" or status == "SELLING":
        # We expect that an order was created before and now we need to check if it still exists or was executed
        # -----
        order_status = await update_order_status()

        order = App.order
        # If order status executed then change the status
        # Status codes: NEW PARTIALLY_FILLED FILLED CANCELED PENDING_CANCEL(currently unused) REJECTED EXPIRED

        if not order or not order_status:
            # No sell order exists or some problem
            # TODO: Recover, reset, init/sync state (cannot trade because wrong happened with the order or connection or whatever)
            #   check connection (like ping), then server status, then our own account status, then funds, orders etc.
            # -----
            await update_trade_status()
            log.error(f"Bad order or order status {order}. Full reset/init needed.")
            return
        if order_status == ORDER_STATUS_FILLED:
            log.info(f"Limit order filled. {order}")
            if status == "BUYING":
                print(f"===> BOUGHT: {order}")
                App.status = "BOUGHT"
            elif status == "SELLING":
                print(f"<=== SOLD: {order}")
                App.status = "SOLD"
            log.info(f'New trade mode: {App.status}')
        elif order_status == ORDER_STATUS_REJECTED or order_status == ORDER_STATUS_EXPIRED or order_status == ORDER_STATUS_CANCELED:
            log.error(f"Failed to fill order with order status {order_status}")
            if status == "BUYING":
                App.status = "SOLD"
            elif status == "SELLING":
                App.status = "BOUGHT"
            log.info(f'New trade mode: {App.status}')
        elif order_status == ORDER_STATUS_PENDING_CANCEL:
            return  # Currently do nothing. Check next time.
        elif order_status == ORDER_STATUS_PARTIALLY_FILLED:
            pass  # Currently do nothing. Check next time.
        elif order_status == ORDER_STATUS_NEW:
            pass  # Wait further for execution
        else:
            pass  # Order still exists and is active
    elif status == "BOUGHT" or status == "SOLD":
        pass  # Do nothing
    else:
        log.error(f"Wrong status value {status}.")

    #
    # Prepare. Kill or update existing orders (if necessary)
    #
    status = App.status

    # If not sold for 1 minute, then kill and then a new order will be created below if there is signal
    # Essentially, this will mean price adjustment (if a new order of the same direction will be created)
    # In future, we might kill only after some timeout
    if status == "BUYING" or status == "SELLING":  # Still not sold for 1 minute
        # -----
        order_status = await cancel_order()
        if not order_status:
            # Cancel exception (the order still exists) or the order was filled and does not exist
            await update_trade_status()
            return
        await asyncio.sleep(1)  # Wait for a second till the balance is updated
        if status == "BUYING":
            App.status = "SOLD"
        elif status == "SELLING":
            App.status = "BOUGHT"

    #
    # Trade by creating orders
    #
    status = App.status

    if signal_side == "BUY":
        print(f"===> BUY SIGNAL {signal}: ")
    elif signal_side == "SELL":
        print(f"<=== SELL SIGNAL: {signal}")
    else:
        print(f"PRICE: {close_price:.2f}")

    # Update account balance etc. what is needed for trade
    # -----
    await update_account_balance()

    if status == "SOLD" and signal_side == "BUY":
        # -----
        try:
            order = await new_limit_order(side=SIDE_BUY, close_time=close_time)
        except OrderLookupUnconfirmed as e:
            # create_order() failed and the follow-up reconciliation lookup could not
            # confirm the true exchange state either. We must not silently retry with
            # a fresh (differently-timed) clientOrderId on the next bar - that could
            # create a second real order on top of one whose fate is unknown. Mark the
            # state as pending BUYING (mirroring a normal in-flight buy order) so the
            # existing BUYING/SELLING reconciliation path at the top of this function
            # (update_order_status()/update_trade_status()) resolves it against the
            # exchange via get_open_orders() before any new order is attempted (B10).
            log.error(f"Order lookup unconfirmed after BUY attempt for {symbol}: {e}. "
                      f"Blocking new submissions until state is confirmed.")
            # App.order may still hold a PREVIOUS, completed order (e.g. the
            # old FILLED SELL right before this BUY attempt). If left as-is,
            # the next cycle's BUYING/SELLING reconciliation would fetch that
            # stale order by its old orderId via update_order_status() and
            # mistake it for the outcome of this unconfirmed attempt. Clear
            # it so the next cycle instead falls back to update_trade_status()
            # (get_open_orders()/account balances), which reflects the real
            # exchange state (B10).
            App.order = None
            App.status = "BUYING"
            return

        if no_trades_only_data_processing:
            print("SKIP TRADING due to 'no_trades_only_data_processing' parameter True")
            # Never change status if orders not executed
        elif order:
            App.status = "BUYING"
        # Never change status if the order was not actually created (e.g. minNotional
        # reject, missing symbol info/filters, missing close price, execute failure)
    elif status == "BOUGHT" and signal_side == "SELL":
        # -----
        try:
            order = await new_limit_order(side=SIDE_SELL, close_time=close_time)
        except OrderLookupUnconfirmed as e:
            # See the matching comment in the BUY branch above (B10).
            log.error(f"Order lookup unconfirmed after SELL attempt for {symbol}: {e}. "
                      f"Blocking new submissions until state is confirmed.")
            # See the matching comment in the BUY branch above (B10): clear
            # App.order so the stale, previously-completed order cannot be
            # mistaken for this unconfirmed attempt on the next cycle.
            App.order = None
            App.status = "SELLING"
            return

        if no_trades_only_data_processing:
            print("SKIP TRADING due to 'no_trades_only_data_processing' parameter True")
            # Never change status if orders not executed
        elif order:
            App.status = "SELLING"
        # Never change status if the order was not actually created (e.g. minNotional
        # reject, missing symbol info/filters, missing close price, execute failure)

    log.info(f"<=== End trade task.")

    return


#
# Order and asset status
#

async def update_trade_status():
    """Read the account state and set the local state parameters."""
    # GET /api/v3/openOrders - get current open orders
    # GET /api/v3/allOrders - get all orders: active, canceled, or filled

    symbol = App.config["symbol"]

    # -----
    try:
        open_orders = collector_binance.client.get_open_orders(symbol=symbol)  # By "open" orders they probably mean "NEW" or "PARTIALLY_FILLED"
        # orders = collector_binance.client.get_all_orders(symbol=symbol, limit=10)
    except Exception as e:
        log.error(f"Binance exception in 'get_open_orders' {e}")
        return

    if not open_orders:
        # -----
        await update_account_balance()

        last_kline = App.analyzer.get_last_kline(symbol)
        last_close_price = to_decimal(last_kline[4])  # Close price of kline has index 4 in the list

        base_quantity = App.account_info.base_quantity  # BTC
        btc_assets_in_usd = base_quantity * last_close_price  # Cost of available BTC in USD

        usd_assets = App.account_info.quote_quantity  # USD

        if usd_assets >= btc_assets_in_usd:
            App.status = "SOLD"
        else:
            App.status = "BOUGHT"

    elif len(open_orders) == 1:
        order = open_orders[0]
        if order.get("side") == SIDE_SELL:
            App.status = "SELLING"
        elif order.get("side") == SIDE_BUY:
            App.status = "BUYING"
        else:
            log.error(f"Neither SELL nor BUY side of the order {order}.")
            return None

    else:  # Many orders
        log.error(f"Wrong state. More than one open order. Fix manually.")
        return None


async def update_order_status():
    """
    Update information about the current order and return its execution status.

    ASSUMPTIONS and notes:
    - Status codes: NEW PARTIALLY_FILLED FILLED CANCELED PENDING_CANCEL(currently unused) REJECTED EXPIRED
    - only one or no orders can be active currently, but in future there can be many orders
    - if no order id(s) is provided then retrieve all existing orders
    """
    symbol = App.config["symbol"]

    # Get currently active order and id (if any)
    order = App.order
    order_id = order.get("orderId", 0) if order else 0
    if not order_id:
        log.error(f"Wrong state or use: check order status cannot find the order id.")
        return None

    # -----
    # Retrieve order from the server
    try:
        new_order = collector_binance.client.get_order(symbol=symbol, orderId=order_id)
    except Exception as e:
        log.error(f"Binance exception in 'get_order' {e}")
        return

    # Impose and overwrite the new order information
    if new_order:
        order.update(new_order)
    else:
        return None

    # Now order["status"] contains the latest status of the order
    return order["status"]


async def update_account_balance():
    """Get available assets (as decimal)."""

    try:
        balance = collector_binance.client.get_asset_balance(asset=App.config["base_asset"])
    except Exception as e:
        log.error(f"Binance exception in 'get_asset_balance' {e}")
        return

    App.account_info.base_quantity = Decimal(balance.get("free", "0.00000000"))  # BTC

    try:
        balance = collector_binance.client.get_asset_balance(asset=App.config["quote_asset"])
    except Exception as e:
        log.error(f"Binance exception in 'get_asset_balance' {e}")
        return

    App.account_info.quote_quantity = Decimal(balance.get("free", "0.00000000"))  # USD

    pass


#
# Cancel and liquidation orders
#

async def cancel_order():
    """
    Kill existing sell order. It is a blocking request, that is, it waits for the end of the operation.
    Info: DELETE /api/v3/order - cancel order
    """
    symbol = App.config["symbol"]

    # Get currently active order and id (if any)
    order = App.order
    order_id = order.get("orderId", 0) if order else 0
    if order_id == 0:
        # TODO: Maybe retrieve all existing (sell, limit) orders
        return None

    # -----
    try:
        log.info(f"Cancelling order id {order_id}")
        new_order = collector_binance.client.cancel_order(symbol=symbol, orderId=order_id)
    except Exception as e:
        log.error(f"Binance exception in 'cancel_order' {e}")
        return None

    # TODO: There is small probability that the order will be filled just before we want to kill it
    #   We need to somehow catch and process this case
    #   If we get an error (say, order does not exist and cannot be killed), then after error returned, we could do trade state reset

    # Impose and overwrite the new order information
    if new_order:
        order.update(new_order)
    else:
        return None

    # Now order["status"] contains the latest status of the order
    return order["status"]


#
# Order creation
#

def _find_filter(filters: list, *filter_types: str) -> Union[dict, None]:
    """Return the first filter dict in `filters` whose filterType is one of `filter_types`."""
    for filter_type in filter_types:
        for f in filters:
            if f.get("filterType") == filter_type:
                return f
    return None


def _decimals_for_step(step: Decimal) -> int:
    """Number of decimal digits implied by an exchange tickSize/stepSize value."""
    exponent = step.normalize().as_tuple().exponent
    return max(-exponent, 0)


def _round_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Round `value` down to the nearest multiple of `step`, using Decimal arithmetic only."""
    value = value if isinstance(value, Decimal) else Decimal(str(value))
    if step <= 0:
        return value
    step = step.normalize()
    quotient = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return (quotient * step).quantize(step, rounding=ROUND_DOWN)


def _generate_client_order_id(symbol: str, side: str, close_time) -> str:
    """
    Deterministic newClientOrderId for one intended order (symbol, side, close_time).

    Binance limits clientOrderId to 36 characters, so we hash the inputs instead of
    concatenating them raw (which could exceed the limit or collide after truncation).
    Same inputs always produce the same id, so a retried submission for the same
    signal (e.g. after a process restart) reuses the same id and can be reconciled
    via get_order(origClientOrderId=...) instead of creating a duplicate order (B09).

    `close_time` must be a real (non-None) value identifying the signal. Falling back
    to a wall-clock timestamp here would make the id different on every call for the
    same signal, defeating idempotency exactly when it is needed most (retries after
    a crash/timeout). Callers must always pass a real close_time (B10).
    """
    if close_time is None:
        raise ValueError(
            "_generate_client_order_id: close_time is required and must not be None "
            "(a non-deterministic fallback would break idempotent order retries)."
        )
    raw = f"{symbol}-{side}-{close_time}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"itb-{digest[:32]}"  # 4 + 32 = 36 chars, within Binance's clientOrderId limit


class OrderLookupUnconfirmed(Exception):
    """
    Raised by `_get_order_by_client_id` when the exchange lookup itself failed (after
    retries) and the true state of the order is unknown - as opposed to a confirmed
    "order does not exist" (Binance code -2013), which is reported as `None`.

    Callers must NOT treat this the same as `None` ("confirmed no order"): they must
    block/reject rather than fall through to normal "no order" handling, because we
    genuinely do not know whether an order exists on the exchange (B10).
    """
    pass


def _get_order_by_client_id(symbol: str, client_order_id: str, max_attempts: int = 3, backoff_seconds: float = 0.5):
    """
    Look up an order by its client-assigned id.

    Used to reconcile state after create_order() raised an exception whose outcome is
    unknown (e.g. a timeout) or reported a duplicate newClientOrderId, so a retry never
    creates a second order (B09).

    Returns the order dict if it exists on the exchange, or None if it is confirmed to
    not exist there (Binance error code -2013).

    Raises `OrderLookupUnconfirmed` if the lookup itself keeps failing (after
    `max_attempts` retries with a short backoff) for any other reason: a different
    Binance error code, a network error, rate limiting, or a generic exception. In
    that case the true state is unknown and callers must not proceed as if there is
    no order (B10).
    """
    if not client_order_id:
        return None

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return collector_binance.client.get_order(symbol=symbol, origClientOrderId=client_order_id)
        except BinanceAPIException as e:
            if getattr(e, "code", None) == -2013:
                # Order does not exist on the exchange: the original create_order() call
                # truly failed. Leave the existing B04 "no order" state.
                log.error(f"No order found for {symbol} with clientOrderId {client_order_id} (code -2013).")
                return None
            last_error = e
            log.error(
                f"Binance exception in 'get_order' while reconciling clientOrderId "
                f"{client_order_id} (attempt {attempt}/{max_attempts}): {e}"
            )
        except Exception as e:
            last_error = e
            log.error(
                f"Exception in 'get_order' while reconciling clientOrderId "
                f"{client_order_id} (attempt {attempt}/{max_attempts}): {e}"
            )

        if attempt < max_attempts:
            time.sleep(backoff_seconds)

    raise OrderLookupUnconfirmed(
        f"Could not confirm order state for {symbol} clientOrderId {client_order_id} "
        f"after {max_attempts} attempts. Last error: {last_error}"
    )


def _reject_order(reason: str) -> None:
    """
    Log a rejection reason and clear App.order/App.order_time so a stale
    (e.g. previously FILLED) order never remains visible in App.order after
    a new order attempt fails. Always returns None so callers can
    `return _reject_order(...)`.
    """
    log.error(reason)
    App.order = None
    App.order_time = now_timestamp()
    return None


async def new_limit_order(side, close_time=None):
    """
    Create a new limit sell order with the amount we current have.
    The amount is total amount and price is determined according to our strategy (either fixed increase or increase depending on the signal).

    `close_time` is the signal's close_time and is used (together with symbol and side)
    to derive a deterministic newClientOrderId, so a retried submission for the same
    signal is idempotent (B09). It is required: a missing close_time is rejected early
    instead of silently falling back to a non-deterministic wall-clock timestamp, which
    would defeat idempotent retries exactly when B09's protection is needed most (B10).

    May raise `OrderLookupUnconfirmed` (propagated from `execute_order`, not caught
    here) if create_order() fails and the exchange reconciliation lookup itself also
    fails after retries. In that case App.order/App.order_time are deliberately left
    untouched (this function returns/raises before updating them), so the caller must
    not assume "no order" and must not submit a new order until the state is confirmed
    on a later cycle (B10).
    """
    symbol = App.config["symbol"]
    now_ts = now_timestamp()

    if close_time is None:
        return _reject_order(
            f"close_time is required to derive a deterministic clientOrderId for {symbol} "
            f"{side} order. Order not submitted."
        )
    client_order_id = _generate_client_order_id(symbol, side, close_time)

    trade_model = App.config.get("trade_model", {})

    #
    # Fetch exchange trading rules (tickSize, stepSize, minNotional) for this symbol.
    # These must be respected instead of a hardcoded rounding precision, otherwise a
    # live order can be rejected by Binance or rounded incorrectly.
    #
    symbol_info = collector_binance.client.get_symbol_info(symbol)
    if not symbol_info:
        return _reject_order(f"Cannot retrieve symbol info for {symbol}. Order not submitted.")

    filters = symbol_info.get("filters", [])
    price_filter = _find_filter(filters, "PRICE_FILTER")
    lot_size_filter = _find_filter(filters, "LOT_SIZE")
    # Binance renamed this filter from MIN_NOTIONAL to NOTIONAL on some symbols.
    notional_filter = _find_filter(filters, "MIN_NOTIONAL", "NOTIONAL")

    if not price_filter or not lot_size_filter:
        return _reject_order(f"Symbol info for {symbol} is missing PRICE_FILTER or LOT_SIZE filter. Order not submitted.")

    tick_size = Decimal(price_filter["tickSize"])
    step_size = Decimal(lot_size_filter["stepSize"])
    min_notional = Decimal(notional_filter["minNotional"]) if notional_filter else None

    #
    # Find limit price (from signal, last kline and adjustment parameters)
    #
    last_kline = App.analyzer.get_last_kline(symbol)
    last_close_price = to_decimal(last_kline[4])  # Close price of kline has index 4 in the list
    if not last_close_price:
        return _reject_order(f"Cannot determine last close price in order to create a market buy order.")

    price_adjustment = trade_model.get("limit_price_adjustment")
    if side == SIDE_BUY:
        price = last_close_price * Decimal(1.0 - price_adjustment)  # Adjust price slightly lower
    elif side == SIDE_SELL:
        price = last_close_price * Decimal(1.0 + price_adjustment)  # Adjust price slightly higher

    price = _round_to_step(price, tick_size)  # Round down to the exchange's tickSize
    price_str = f"{price:.{_decimals_for_step(tick_size)}f}"

    #
    # Find quantity
    #
    if side == SIDE_BUY:
        # Find how much quantity we can buy for all available USD using the computed price
        quantity = App.account_info.quote_quantity  # USD
        percentage_used_for_trade = trade_model.get("percentage_used_for_trade")
        quantity = (quantity * percentage_used_for_trade) / Decimal(100.0)  # Available for trade
        quantity = quantity / price  # BTC to buy
        # Alternatively, we can pass quoteOrderQty in USDT (how much I want to spend)
    elif side == SIDE_SELL:
        # All available BTCs
        quantity = App.account_info.base_quantity  # BTC

    quantity = _round_to_step(quantity, step_size)  # Round down to the exchange's stepSize
    quantity_str = f"{quantity:.{_decimals_for_step(step_size)}f}"

    #
    # Reject orders that do not meet the exchange's minimum notional value instead
    # of letting Binance reject them (or silently mis-rounding them).
    #
    if min_notional is not None and price * quantity < min_notional:
        return _reject_order(
            f"Order notional {price * quantity} for {symbol} is below the exchange minNotional "
            f"{min_notional}. Order not submitted."
        )

    #
    # Execute order
    #
    order_spec = dict(
        symbol=symbol,
        side=side,
        type=ORDER_TYPE_LIMIT,  # Alternatively, ORDER_TYPE_LIMIT_MAKER
        timeInForce=TIME_IN_FORCE_GTC,
        quantity=quantity_str,
        price=price_str,
        newClientOrderId=client_order_id,
    )

    if trade_model.get("no_trades_only_data_processing"):
        print(f"NOT executed order spec: {order_spec}")
        order = None
    else:
        order = execute_order(order_spec)

    #
    # Store/log order object in our records (only after confirmation of success)
    #
    App.order = order
    App.order_time = now_ts

    return order


def execute_order(order: dict):
    """
    Validate and submit order.

    Raises `OrderLookupUnconfirmed` (propagated from `_get_order_by_client_id`, not
    caught here) if create_order() fails and the follow-up reconciliation lookup also
    fails to confirm the order's true state on the exchange. This is intentional: the
    caller (`new_limit_order`) must not treat an unconfirmed lookup as "no order" and
    must not overwrite App.order/App.status, so a new (possibly duplicate) order is
    never submitted while the state is unknown (B10).
    """

    trade_model = App.config.get("trade_model", {})

    # TODO: Check validity, e.g., against filters (min, max) and our own limits

    if trade_model.get("test_order_before_submit"):
        try:
            log.info(f"Submitting test order: {order}")
            test_response = collector_binance.client.create_test_order(**order)  # Returns {} if ok. Does not check available balances - only trade rules
        except Exception as e:
            log.error(f"Binance exception in 'create_test_order' {e}")
            # TODO: Reset/resync whole account
            return

    if trade_model.get("simulate_order_execution"):
        # TODO: Simply store order so that later we can check conditions of its execution
        print(order)
        pass
    else:
        # -----
        # Submit order
        symbol = order.get("symbol")
        client_order_id = order.get("newClientOrderId")
        try:
            log.info(f"Submitting order: {order}")
            order = collector_binance.client.create_order(**order)
        except Exception as e:
            # The outcome of create_order() is unknown here: it may have failed before
            # reaching the exchange, or it may have succeeded while the response was
            # lost (e.g. a timeout), or Binance may have rejected it as a duplicate
            # newClientOrderId (code -2010) because a previous attempt already went
            # through. In all these cases we must NOT assume failure and silently
            # retry (that could create a duplicate order); instead reconcile with the
            # exchange using the same deterministic clientOrderId (B09).
            code = getattr(e, "code", None)
            if code == -2010:
                log.warning(
                    f"Duplicate newClientOrderId {client_order_id} for {symbol}: "
                    f"reconciling with the existing order instead of resubmitting."
                )
            else:
                log.error(
                    f"Binance exception in 'create_order' {e}. Reconciling with the "
                    f"exchange via clientOrderId {client_order_id} before giving up."
                )
            return _get_order_by_client_id(symbol, client_order_id)

        if not order or not order.get("status"):
            return None

    return order
