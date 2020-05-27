##python3 -m venv my-project-env to create env
## source env/bin/activate

from datetime import datetime
import pandas as pd
from pandas import DataFrame as df
from scrapping import keys
from binance.client import Client
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def binance_price():
    client = Client(api_key=keys.Pkey, api_secret=keys.Skey)

    candles1 = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, limit=1000)

    candles1_data_frame = df(candles1)
    candles1_data_frame_date = candles1_data_frame[0]
    final_date1 = []
    for time in candles1_data_frame_date.unique():
        date_clair = datetime.fromtimestamp(int(time/1000))
        final_date1.append(date_clair)
    dataframe_final_date1 = df(final_date1)
    dataframe_final_date1.columns =  ['date']

    final_dataframe1= candles1_data_frame.join(dataframe_final_date1)
    final_dataframe1.set_index('date', inplace=True)
    final_dataframe1.columns = ['timestamp','open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trade_number', 'taker_buy_base', 'taker_buy_quote','ignore']
    del final_dataframe1['ignore']
    final_dataframe1["open"] = pd.to_numeric(final_dataframe1["open"])
    final_dataframe1["high"] = pd.to_numeric(final_dataframe1["high"])
    final_dataframe1["low"] = pd.to_numeric(final_dataframe1["low"])
    final_dataframe1["close"] = pd.to_numeric(final_dataframe1["close"])
    final_dataframe1["volume"] = pd.to_numeric(final_dataframe1["volume"])
    final_dataframe1["close_time"] = pd.to_numeric(final_dataframe1["close_time"])
    final_dataframe1["quote_asset_volume"] = pd.to_numeric(final_dataframe1["quote_asset_volume"])
    final_dataframe1["trade_number"] = pd.to_numeric(final_dataframe1["trade_number"])
    final_dataframe1["taker_buy_base"] = pd.to_numeric(final_dataframe1["taker_buy_base"])
    final_dataframe1["taker_buy_quote"] = pd.to_numeric(final_dataframe1["taker_buy_quote"])


    df['res'] = df['Open'] / df['Close']
    return final_dataframe1


print(binance_price())

