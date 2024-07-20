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
from binance.client import Client
import pickle

# Define the Binance API
load_dotenv()

api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")

binance_api = Client(api_key, api_secret)

# Test borrowing APIs
main_asset = "ETH"
symbol = "ETHUSDT"

isolated_margin_account = binance_api.get_isolated_margin_account(
    symbols=symbol)

borrow_amount = '{:.4f}'.format(
    12/float(isolated_margin_account['assets'][0]["indexPrice"]))

# binance_api.create_margin_loan(
#     asset=main_asset, amount=borrow_amount, isIsolated='TRUE', symbol=symbol)

# print("Loan taken")
# time.sleep(10)

# binance_api.create_margin_order(
#     symbol=symbol, isIsolated='TRUE', side="SELL", type="MARKET", quantity=borrow_amount
# )
# print("Short sell done")

# binance_api.repay_margin_loan(
#     asset=main_asset, amount=borrow_amount, isIsolated='TRUE', symbol=symbol)
print(borrow_amount)
binance_api.create_margin_order(
    symbol=symbol, isIsolated='TRUE', side="BUY", type="MARKET", sideEffectType="AUTO_REPAY", quantity=borrow_amount
)
