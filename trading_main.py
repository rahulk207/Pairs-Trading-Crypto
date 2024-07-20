import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import websocket
import threading
from datetime import datetime
from dotenv import load_dotenv
import os
import logging
import time
from binance.client import AsyncClient
import pickle
import asyncio

from pykalman import KalmanFilter
from trading_strategy_utils import *
from binance_api_functions import *

logging.basicConfig(filename='trading.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add console handler to logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)


async def fetchPriceAndTrade():
    global data_btc
    global data_eth

    while True:
        # Fetch price of BTC and assign to data_btc
        data_btc = await binance_api_with_retry(get_symbol_ticker, binance_api, symbol="BTCUSDT")
        data_eth = await binance_api_with_retry(get_symbol_ticker, binance_api, symbol="ETHUSDT")

        # print(price_data)

        # data_btc = price_data[0]
        # data_eth = price_data[1]

        logger.info("Data BTC: {}".format(json.dumps(data_btc)))
        logger.info("Data ETH: {}".format(json.dumps(data_eth)))

        await checkTradingLogic()

        await asyncio.sleep(60)  # Wait for 1 minute


async def shortMarginTrading(collateral_level, trade_dollars_amount, symbol, main_asset):
    # Check the isolated margin account balance
    isolated_margin_account = await binance_api_with_retry(get_isolated_margin_account, binance_api, symbols=symbol)
    current_collateral = float(
        isolated_margin_account['assets'][0]['quoteAsset']['totalAsset'])

    if current_collateral < collateral_level:
        # Add collateral to the isolated margin account
        transfer_margin_to_isolated = await binance_api_with_retry(transfer_spot_to_isolated_margin, binance_api, transfer_amount=collateral_level-current_collateral, symbol=symbol)
        logger.info("Transfer to margin amount: {}".format(
            json.dumps(transfer_margin_to_isolated)))
        # Send an alert here

    # Borrow ETH for shorting in isolated margin mode
    borrow_amount = '{:.4f}'.format(trade_dollars_amount /
                                    float(isolated_margin_account['assets'][0]["indexPrice"]))
    loan_response = await binance_api_with_retry(create_margin_loan, binance_api, asset=main_asset, borrow_amount=borrow_amount, symbol=symbol)
    logger.info("Loan response: {}".format(json.dumps(loan_response)))

    loan_conversion_to_usd_response = await binance_api_with_retry(create_margin_order, binance_api, borrow_amount=borrow_amount, symbol=symbol, side="SELL", type="MARKET")
    logger.info("Loan conversion to USDT response: {}".format(
        json.dumps(loan_conversion_to_usd_response)))

    return float(borrow_amount)


async def long(amount, symbol):
    long_response = await binance_api_with_retry(order_market_buy, binance_api, amount=float('{:.4f}'.format(amount)), symbol=symbol)
    logger.info("Long order response: {}".format(json.dumps(long_response)))


async def openPosition(current_state):
    global trading_states

    # Define the collateral level and the amount of ETH and BTC to trade
    collateral_level = 5  # This can be adjusted based on your preference
    trade_dollars_amount = 10

    # Get spot account balance for USDT
    spot_account_balance = await binance_api_with_retry(get_asset_balance, binance_api, asset="USDT")
    logger.info(
        "Spot account balance for USDT before opening position: {}".format(json.dumps(spot_account_balance)))

    if current_state == 2:
        # Above line Open. short ETH, long BTC
        trading_states["position"] = 1

        borrow_amount = await shortMarginTrading(
            collateral_level, trade_dollars_amount, "ETHUSDT", "ETH")

        # Buy BTC using spot trading
        long_amount = trading_states["reg_slope"]*borrow_amount
        await long(long_amount, "BTCUSDT")

    elif current_state == -2:
        # Below line Open. short BTC, long ETH
        trading_states["position"] = -1

        borrow_amount = await shortMarginTrading(
            collateral_level, trade_dollars_amount, "BTCUSDT", "BTC")

        # Buy BTC using spot trading
        long_amount = borrow_amount/trading_states["reg_slope"]
        await long(long_amount, "ETHUSDT")

    trading_states["is_open"] = True
    trading_states["open_slope"] = trading_states["reg_slope"]

    trading_states["borrow_amount"] = borrow_amount
    trading_states["long_amount"] = long_amount

    # Now open the positions. Also log price of entry. And transaction cost.


async def repayMarginTrading(borrow_amount, symbol):
    # Repay the loan in isolated margin mode
    repay_loan_response = await binance_api_with_retry(create_margin_order, binance_api, borrow_amount=borrow_amount, symbol=symbol, side="BUY", type="MARKET", sideEffectType="AUTO_REPAY")
    logger.info("Repay loan response: {}".format(
        json.dumps(repay_loan_response)))


async def closeLong(amount, symbol):
    close_long_response = await binance_api_with_retry(order_market_sell, binance_api, amount=float('{:.4f}'.format(amount)), symbol=symbol)
    logger.info("Close long position response: {}".format(
        json.dumps(close_long_response)))


async def closePosition():
    global trading_states

    # After repaying, ideally everything should be converted to USDT
    if trading_states["position"] == 1:
        isolated_margin_account = await binance_api_with_retry(get_isolated_margin_account, binance_api, symbols="ETHUSDT")
        logger.info("Isolated account before closing ETHUSDT: {}".format(
            isolated_margin_account))

        await repayMarginTrading(trading_states["borrow_amount"], "ETHUSDT")
        await closeLong(trading_states["long_amount"], "BTCUSDT")

        isolated_margin_account = await binance_api_with_retry(get_isolated_margin_account, binance_api, symbols="ETHUSDT")
        transfer_margin_to_spot = await binance_api_with_retry(transfer_isolated_margin_to_spot, binance_api, transfer_amount=isolated_margin_account['assets'][0]['quoteAsset']['free'], symbol='ETHUSDT')
        logger.info("Transferred balance from margin to spot ETHUSDT: {}".format(
            json.dumps(transfer_margin_to_spot)))

    elif trading_states["position"] == -1:
        await repayMarginTrading(trading_states["borrow_amount"], "BTCUSDT")
        await closeLong(trading_states["long_amount"], "ETHUSDT")
        isolated_margin_account = await binance_api_with_retry(get_isolated_margin_account, binance_api, symbols="BTCUSDT")
        transfer_margin_to_spot = await binance_api_with_retry(transfer_isolated_margin_to_spot, binance_api, transfer_amount=isolated_margin_account['assets'][0]['quoteAsset']['totalAsset'], symbol='BTCUSDT')
        logger.info("Transferred balance from margin to spot BTCUSDT: {}".format(
            json.dumps(transfer_margin_to_spot)))

    spot_account_balance = await binance_api_with_retry(get_asset_balance, binance_api, asset="USDT")
    logger.info("Spot account balance for USDT after closing position: {}".format(
        json.dumps(spot_account_balance)))

    trading_states["close_durations"].append(trading_states["open_time"])
    trading_states["position"] = 0
    trading_states["is_open"] = False
    trading_states["open_time"] = 0
    trading_states["cum_slope_diff"] = 0


async def checkTradingLogic():
    global data_btc
    global data_eth
    global trading_states

    if data_btc and data_eth:
        print("Current position", trading_states["position"])

        price_btc = float(data_btc['price'])
        price_eth = float(data_eth['price'])

        trading_states["kalman_theta"], trading_states["kalman_P"] = kalmanUpdate(
            np.array(price_btc).reshape(1, 1), np.array(price_eth).reshape(1, 1), trading_states["kalman_theta"], trading_states["kalman_P"], kalman_params["W"], kalman_params["sigma_e"], kalman_params["window"])
        print("New slope", trading_states["kalman_theta"][0][0])

        trading_states["prev_slope"] = trading_states["reg_slope"]
        trading_states["reg_slope"] = trading_states["kalman_theta"][0][0]
        print("Open slope", trading_states["open_slope"])

        if trading_states["is_open"]:
            trading_states["cum_slope_diff"] += (
                trading_states["reg_slope"] - trading_states["prev_slope"])
            trading_states["open_time"] += 1
            if np.abs(trading_states["cum_slope_diff"]) > trading_params["slope_diff_threshold"]:
                await closePosition()
                # Send an alert here and stop trading. Exit code.

            current_state = currentPos(
                price_btc, price_eth, trading_states["open_slope"], trading_params["lower_bound"], trading_params["upper_bound"])

        else:
            current_state = currentPos(
                price_btc, price_eth, trading_states["reg_slope"], trading_params["lower_bound"], trading_params["upper_bound"])

        print("Current state", current_state)

        if (current_state == 2 or current_state == -2) and trading_states["position"] == 0:
            await openPosition(current_state)
            #  Send an alert message here as well.
        elif (current_state == -1 or current_state == -2) and trading_states["position"] == 1:
            await closePosition()
            #  Send an alert message here as well.
        elif (current_state == 1 or current_state == 2) and trading_states["position"] == -1:
            await closePosition()
            #  Send an alert message here as well.

        data_btc = None
        data_eth = None

    # Save trading_states here. Also modify code to pick-up latest states when code is restarted.
    with open('trading_states.pickle', 'wb') as file:
        pickle.dump(trading_states, file)


async def main():
    global binance_api
    binance_api = await AsyncClient.create(api_key, api_secret)
    await fetchPriceAndTrade()

if __name__ == "__main__":
    data_btc = None
    data_eth = None

    kalman_params = {
        "W": np.array([1e-6]).reshape(1, 1),
        "sigma_e": 3,
        "window": 1
    }

    trading_params = {
        "slope_diff_threshold": 0.02,
        "lower_bound": -0.05,
        "upper_bound": 0.05,
    }

    if os.path.exists('trading_states.pickle'):
        with open('trading_states.pickle', 'rb') as file:
            trading_states = pickle.load(file)
    else:
        trading_states = {
            "kalman_theta": np.array([0.02]).reshape(1, 1),
            "kalman_P": np.array([1e-6]).reshape(1, 1),
            "position": 0,
            "returns": [],
            "percentage_returns": [],
            "transaction_costs": [],
            "is_open": False,
            "borrow_amount": 0,
            "long_amount": 0,
            "open_time": 0,
            "close_durations": [],
            "cum_slope_diff": 0,
            "open_slope": 0,
            "prev_slope": 0,
            "reg_slope": 0
        }

    # Define the Binance API
    load_dotenv()

    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
