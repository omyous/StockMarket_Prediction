from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import LSTM, Dense
from tensorflow_core.python.keras.losses import mean_squared_error

import src.scrapping.tweets as tweeter
import src.scrapping.stock_prices as yahoo
import pandas as pd
import numpy as np


class Custom_dataset():
    def __init__(self):
        self.tweets_path = "data/google_tweets_.csv"
        self.prices_path = "data/google.csv"
        self.tweets = None
        self.prices = None

    def get_data(self):
        self.tweets = pd.read_csv(self.tweets_path)
        self.prices = pd.read_csv(self.prices_path)


    def clean_data(self):
        self.get_data()
        self.clean_tweets()
        self.clean_prices()



    def clean_tweets(self):

        self.tweets["date"] = pd.to_datetime(self.tweets["date"], format='%Y/%m/%d')
        self.tweets["date"] = self.tweets["date"].dt.date
        #drop any duplicate post
        self.tweets = self.tweets.drop_duplicates()
        self.tweets['expand'] = self.tweets.apply(lambda x: '. '.join([x['text']]), axis=1)
        self.tweets = self.tweets.groupby('date')['expand'].apply(list)
        self.tweets = pd.DataFrame(data=self.tweets.values, index=self.tweets.index, columns=["text"])
        #create one daily tweet  intead of many ones
        text = [' '.join(sentence) for sentence in self.tweets["text"]]
        self.tweets["text"] = text
        self.tweets.to_csv("data/clean_tweets.csv", index=False)

    def clean_prices(self):

        #delete zeros columns
        self.prices = self.prices.loc[:, (self.prices != 0).any(axis=0)]
        self.prices["Date"] = pd.to_datetime(self.prices["Date"], format='%Y/%m/%d')
        self.prices["Date"] = self.prices["Date"].dt.date
        # drop duplicate
        self.prices = self.prices.drop_duplicates(subset='Date', keep="last")
        self.prices.to_csv("data/clean_prices.csv", index=False)

    def merge_data(self):
        tweets = pd.read_csv("data/tweets_scores.csv")
        prices = pd.read_csv("data/clean_prices.csv")
        print()
        tweets_dates = list(tweets["date"])
        prices_dates = list(prices["Date"])
        l=[]
        for i in prices_dates:
            if not i in tweets_dates:
                print(i)
                tweets = tweets.append({'date': i,
                                        'positive': tweets["positive"].median(),
                                        'negative':tweets['negative'].median()
                                        },
                                       ignore_index=True)

        print("####################################\n\n")

        for i in tweets_dates:
            if not i in prices_dates:
                prices = prices.append({'Date': i,
                                        'High': prices["High"].median(),
                                        'Low':prices['Low'].median(),
                                        'Open':prices['Open'].median(),
                                        'Close':prices['Close'].median(),
                                        'Volume':prices['Volume'].median(),
                                        'Adj Close':prices['Adj Close'].median()
                                        },
                                       ignore_index=True
                                       )

        df = tweets.rename(columns={'date': 'Date'})
        df = prices.merge(df, on="Date")
        tweets.sort_values("date").to_csv("data/tweets_scores_.csv", index=False)
        prices.sort_values("Date").to_csv("data/clean_prices_.csv", index=False)
        df.to_csv("data/merged.csv", index=False)

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

import matplotlib.pyplot as plt
def method():

    dataset = pd.read_csv('data/merged.csv', header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    print(reframed.shape)
    # split into train and test sets
    values = reframed.values
    prop = int(values.shape[0]*.8)
    train = values[:prop, :]
    test = values[prop:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-3], train[:, 3]
    test_X, test_y = test[:, :-3], test[:, 3]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=500, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)





if __name__ == "__main__":
    #df = Custom_dataset()
    #df.clean_data()
    #df.merge_data()

    method()
    """"  t = df.tweets
    t["date"] = pd.to_datetime(t["date"], format='%Y/%m/%d')

    t["date"]=t["date"].dt.date
    t['expand'] = t.apply(lambda x: ', '.join([x['text']]), axis=1)
    t=t.groupby('date')['expand'].apply(list)
    t = pd.DataFrame(data=t.values, index=t.index, columns=["text"])
    print(t.head())
    """



