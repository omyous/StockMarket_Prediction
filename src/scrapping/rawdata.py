##python3 -m venv my-project-env to create env
## source env/bin/activate



from datetime import datetime

from binance.client import Client
from pandas import DataFrame as df, DataFrame

from scrapping import keys

symbol_btcusdt = 'BTCUSDT'


def getrawdata(symbol, start, end):
    client = Client(api_key=keys.Pkey, api_secret=keys.Skey)
    candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=1000, startTime=start,
                                endTime=end)
    candles_data_frame: DataFrame = df(candles)
    return candles_data_frame


def format_date(data_frame):
    candles_data_frame_date = data_frame[0]
    final_date = []
    for time in candles_data_frame_date.unique():
        date_clair = datetime.fromtimestamp(int(time/1000))
        final_date.append(date_clair)

    data_frame.pop(0)
    data_frame.pop(11)
    dataframe_final_date = df(final_date)
    dataframe_final_date.columns = ['date']

    final_dataframe = data_frame.join(dataframe_final_date)
    final_dataframe.set_index('date', inplace=True)
    final_dataframe.columns = ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                               'trade_number', 'taker_buy_base', 'taker_buy_quote']
    return final_dataframe


#print(format_date(getrawdata('BTCUSDT', 1583035200000, 1585713600000)))


def get_data_2000hours(symbol, before_date):
    before_date_minus_1000 = before_date - 3600000000
    before_date_minus_2000 = before_date_minus_1000 - 3600000000


    data_frame_1000 = getrawdata(symbol, before_date_minus_1000 , before_date)
    data_frame_2000 = getrawdata(symbol, before_date_minus_2000 , before_date_minus_1000)

    #data_frame_2000hours = concat([data_frame_1000,data_frame_2000])
    #result= data_frame_1000.append(data_frame_2000)
    #result.drop_duplicates(['date'])
    formated_data_2000hours = format_date(data_frame_2000)
    formated_data_1000hours = format_date(data_frame_1000)
    data_finaal = formated_data_2000hours.append(formated_data_1000hours)
    return data_finaal


"""pd.set_option('display.max_rows', None)  #to show all the dataframe lines and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)"""
#print(print_data_2000hours(symbol_btcusdt, 1584849600000))


po = (get_data_2000hours(symbol_btcusdt, 1588674497000))
print(po)
"""ouv = po['open']
fer = po['close']"""

"""print(ouv.values)"""

"""for i in po.index:
    float(po['open'][i])"""



#resultat = []
"""try:
    for i in po.index:
       resultat.append((float((po['open'][i])) - float((po['close'][i]))))
except TypeError:
    print()"""



"""print((resultat.dtypes))"""

"""res = po.assign(resultat=[])
po['resultat']= ouv - fer
print(po.head(3))"""