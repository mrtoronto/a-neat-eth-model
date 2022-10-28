import requests
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime

def _get_binance_data(start_date, end_date, symbol='ETHUSDT', interval='1h', futures=False):
    if futures:
        url = "https://fapi.binance.com/fapi/v1/klines"
    else:
        url = "https://api.binance.com/api/v3/klines"
    
    start_time = 0
    all_rows = []
    end_time = datetime(end_date[0],end_date[1],end_date[2]).timestamp()
    start_time = datetime(start_date[0], start_date[1], start_date[2]).timestamp()

    while start_time <= end_time:
        rows = json.loads(requests.get(url, params={'symbol': symbol, 'startTime': int(start_time) * 1000, 'interval': interval}).text)
        all_rows += rows
        start_time = max([int(r[0] / 1000) for r in all_rows])

    all_rows = [[datetime.fromtimestamp(r[0] / 1000).strftime("%Y-%m-%d %H:%M:%S"), r[1]] for r in all_rows]
    data = pd.DataFrame(all_rows, columns=['datetime', 'price'])
    data['price'] = data['price'].astype(float)
    return data


def rsi(prices, periods = 14, ema = True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = prices.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

def get_bollinger_bands(prices, rate=20):
    sma = prices.rolling(rate).mean()
    std = prices.rolling(rate).std()
    bollinger_up = sma + std * 2 # Calculate top band
    bollinger_down = sma - std * 2 # Calculate bottom band
    return bollinger_up, bollinger_down

def _add_indicators(data):

    for n in [3, 5, 10, 25, 50, 100, 200, 300]:
        weights = np.arange(1,n+1) #this creates an array with integers 1 to 10 included
        data[f'SMA {n}'] = data['price'].rolling(n).mean()
        data[f'WMA {n}'] = data['price'].rolling(n).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
        data[f'EMA {n}'] = data['price'].ewm(span=n).mean()

    for n in [6, 12, 24, 48, 168, 336]:
        data[f'rsi {n}'] = rsi(data['price'], n)


    for n in [20, 40, 60]:
        bollinger_up, bollinger_down = get_bollinger_bands(data['price'], rate=n)
        data[f'boll_up_{n}'] = bollinger_up
        data[f'boll_down_{n}'] = bollinger_down

    return data