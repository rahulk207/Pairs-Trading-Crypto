from binance.client import AsyncClient
import asyncio

MAX_RETRIES = 5


async def binance_api_with_retry(api_function, binance_client, **kwargs):
    for retry in range(MAX_RETRIES + 1):
        try:
            result = await api_function(binance_client, **kwargs)
            return result  # If the API call is successful, return the result.
        except Exception as e:
            if retry < MAX_RETRIES:
                print(f'API call failed (Retry {retry + 1}/{MAX_RETRIES})')

            else:
                print('API call failed after max retries.')
                raise e  # If all retries fail, re-raise the error.


async def get_symbol_ticker(binance_client, symbol):
    data = await binance_client.get_symbol_ticker(symbol=symbol)
    return data


async def get_isolated_margin_account(binance_client, symbols):
    data = await binance_client.get_isolated_margin_account(symbols=symbols)
    return data


async def get_asset_balance(binance_client, asset):
    data = await binance_client.get_asset_balance(asset=asset)
    return data


async def transfer_spot_to_isolated_margin(binance_client, transfer_amount, symbol):
    data = await binance_client.transfer_spot_to_isolated_margin(asset='USDT', symbol=symbol, amount=transfer_amount)
    return data


async def transfer_isolated_margin_to_spot(binance_client, transfer_amount, symbol):
    data = await binance_client.transfer_isolated_margin_to_spot(asset='USDT', symbol=symbol, amount=transfer_amount)
    return data


async def create_margin_loan(binance_client, asset, borrow_amount, symbol):
    data = await binance_client.create_margin_loan(asset=asset, amount=borrow_amount, isIsolated='TRUE', symbol=symbol)
    return data


async def create_margin_order(binance_client, borrow_amount, symbol, side, type, sideEffectType="NO_SIDE_EFFECT"):
    data = await binance_client.create_margin_order(symbol=symbol, isIsolated='TRUE', side=side, type=type, sideEffectType=sideEffectType, quantity=borrow_amount)
    return data


async def order_market_buy(binance_client, amount, symbol):
    data = await binance_client.order_market_buy(symbol=symbol, quantity=amount)
    return data


async def order_market_sell(binance_client, amount, symbol):
    data = await binance_client.order_market_sell(symbol=symbol, quantity=amount)
    return data
