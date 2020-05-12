import math
from datetime import datetime
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import keys
from binance.client import Client

client = Client(api_key=keys.Pkey,api_secret=keys.Skey)


candles2 = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, limit=1000, startTime=22, endTime=22)

candles2_data_frame = df(candles2)
candles2_data_frame_date = candles2_data_frame[0]
final_date2 = []
for time in candles2_data_frame_date.unique():
    date_clair = datetime.fromtimestamp(int(time/1000))
    final_date2.append(date_clair)

candles2_data_frame.pop(0)
candles2_data_frame.pop(11)
dataframe_final_date2 = df(final_date2)
dataframe_final_date2.columns =  ['date']

    final_dataframe2= candles2_data_frame.join(dataframe_final_date2)
    final_dataframe2.set_index('date', inplace=True)
    final_dataframe2.columns = ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trade_number', 'taker_buy_base', 'taker_buy_quote']