import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
class LSTM_data():
    def __init__(self):
        self.raw_data = pd.read_csv("data/merged.csv",
                                    index_col=0
                                    )

        self.add_noise()
        self.lag: int = 5
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.get_memory()

    def get_XY(self):
        data = self.raw_data
        Y = data[["Close"]]
        Y.columns = ["Y"]
        # shift the features
        cols = ["High", "Low", "Open", "Volume", "Adj Close", "positive", "negative"]  #
        dres = data
        lag_value = 5

        # shift to get all the data from X
        for i in range(1, lag_value + 1):
            dtemp = data[cols].shift(periods=i)
            dtemp.columns = [c + '_lag' + str(i) for c in cols]
            dres = dres.merge(dtemp, on='Date')
        X = dres

        # shift the dependant variable
        cols = Y.columns
        dy = pd.DataFrame(index=Y.index)
        for i in range(1, lag_value + 1):
            dytemps = Y.shift(periods=i)
            dytemps.columns = [c + '_lag' + str(i) for c in cols]
            dy = dy.merge(dytemps, on='Date')
        X = X.merge(dy, on='Date')
        X = X.dropna()


        X.columns = ['High_lag1', 'Low_lag1', 'Open_lag1', 'Volulag1', 'Adj Close_lag1',
                     'High_lag2', 'Low_lag2', 'Open_lag2', 'Volume_lag2', 'Adj Close_lag2',
                     'High_lag3', 'Low_lag3', 'Open_lag3', 'Volume_lag3', 'Adj Close_lag3',
                     'High_lag4', 'Low_lag4', 'Open_lag4', 'Volume_lag4', 'Adj Close_lag4',
                     'High_lag5', 'Low_lag5', 'Open_lag5', 'Volume_lag5', 'Adj Close_lag5',
                     'Y_lag1', 'Y_lag2', 'Y_lag3', 'Y_lag4', 'Y_lag5',
                     'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close',
                     'positive', 'negative',
                     'positive_lag1', 'negative_lag1',
                     'positive_lag2', 'negative_lag2',
                     'positive_lag3', 'negative_lag3',
                     'positive_lag4', 'negative_lag4',
                     'positive_lag5', 'negative_lag5'
                     ]
        Y = X[["Close"]].values
        self.train_dates = X.index.values[:int(X.shape[0] * .8)]
        self.test_dates = X.index.values[int(X.shape[0] * .8):]
        X = X.drop(["Close",
                    "Adj Close",
                    'Adj Close_lag1',
                    'Adj Close_lag2',
                    'Adj Close_lag3',
                    'Adj Close_lag4',
                    'Adj Close_lag5'],
                   axis=1).values
        return X, Y

    def add_noise(self):
        mu, sigma = 0, 0.1
        # creating a noise with the same dimension as the dataset (2,2)
        noise = np.random.normal(mu, sigma, self.raw_data[["positive", "negative"]].shape)
        self.raw_data[["positive", "negative"]] = np.add(self.raw_data[["positive", "negative"]].values, noise)

    def lag_func(self, data):
        xs = []
        ys = []

        for i in range(len(data) - self.lag - 1):
            x = data[i:(i + self.lag)]
            y = data[i + self.lag, 0]
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def get_splited_data(self):
        X, Y = self.get_XY()

        # split the data to test and train sets
        X_train, X_test = X[:int(X.shape[0] * .8), :], X[int(X.shape[0] * .8):, :]
        y_train, y_test = Y[:int(X.shape[0] * .8), :], Y[int(X.shape[0] * .8):, :]

        # normalize the explicative variables
        X_train_scaled = self.x_scaler.fit_transform(X_train)
        X_test_scaled = self.x_scaler.transform(X_test)

        # normalize the dependant variable
        Y_train_scaled = self.y_scaler.fit_transform(y_train)
        Y_test_scaled = self.y_scaler.transform(y_test)
        return X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled

    def get_memory(self):
        X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled = self.get_splited_data()

        # convert the 2D datasets to to 3D datasets
        X_train_reshaped = np.zeros(shape=(int(X_train_scaled.shape[0] / 1), 1, X_train_scaled.shape[1]),
                                    dtype=np.float32)
        X_test_reshaped = np.zeros(shape=(int(X_test_scaled.shape[0] / 1), 1, X_train_scaled.shape[1]),
                                   dtype=np.float32)
        for i in range(X_train_reshaped.shape[0]):
            X_train_reshaped[i] = X_train_scaled[i:i + 1, :]

        for i in range(X_test_reshaped.shape[0]):
            X_test_reshaped[i] = X_test_scaled[i:i + 1, :]

        return X_train_reshaped, X_test_reshaped, Y_train_scaled, Y_test_scaled


