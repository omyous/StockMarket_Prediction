import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

from machine_learning.sentiments_analysis import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 999

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



    def get_vectors(self,strs):
        text = [t for t in strs]
        vectorizer = CountVectorizer(text)
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()

    def get_cosine_sim(self,strs):
        vectors = [t for t in self.get_vectors(strs)]
        return cosine_similarity(vectors)


    def tweets_sim(self,df):
        df_ = df
        print(df_)
        # create a clean dataframe
        clean_df2 = pd.DataFrame(columns=["date", "text"])
        indexes = df_.index.values.tolist()
        i = 0
        while i < len(indexes):
            if df_.loc[indexes[i]]["text"] != "str":
                j = i + 1
                while j < len(indexes):
                    if df_.loc[indexes[j]]["text"] != "str":
                        df_.loc[indexes[i]]["text"] = " ".join(sent_tokenize(df_.loc[indexes[i]]["text"]))
                        df_.loc[indexes[j]]["text"] = " ".join(sent_tokenize(df_.loc[indexes[j]]["text"]))

                        result = self.get_cosine_sim(df_.iloc[[i, j]]["text"])

                        if result[0, 1] >= .5:
                            print(result, "\n")
                            print("SIMILAIRE: ", df_.iloc[[i, j]]["text"])
                        else:
                            print("DIFFERENT: ", df_.iloc[[i, j]]["text"])
                            print(result)
                            clean_df2 = clean_df2.append({'date': df.loc[indexes[j]]["date"],
                                                        'text': df.loc[indexes[j]]["text"]},
                                                       ignore_index=True)

                    df_.loc[indexes[j]]["text"] = "str"
                    j += 1
                clean_df2 = clean_df2.append({'date': df.loc[indexes[i]]["date"],
                                            'text': df.loc[indexes[i]]["text"]},
                                           ignore_index=True)
                df_.loc[indexes[i]]["text"] = "str"
            i += 1

        return clean_df2

    def drop_tweetsduplicates(self):
        dates = self.tweets["date"].unique()
        clean_df1 = pd.DataFrame(columns=["date", "text"])
        for d in dates:
            df_ = self.tweets[self.tweets["date"] == d]
            clean_df1 = clean_df1.append(self.tweets_sim(df_))
        return clean_df1

    def clean_tweets(self):

        self.tweets["date"] = pd.to_datetime(self.tweets["date"], format='%Y/%m/%d').dt.date

        self.tweets = self.drop_tweetsduplicates()
        #drop any duplicate post
        self.tweets = self.tweets.drop_duplicates("text")
        self.tweets['expand'] = self.tweets.apply(lambda x: '. '.join([x['text']]), axis=1)
        self.tweets = self.tweets.groupby('date')['expand'].apply(list)
        #self.tweets["date"] = self.tweets.index.values
        self.tweets = pd.DataFrame(data=self.tweets.values, index=self.tweets.index, columns=["text"])
        #create one daily tweet  intead of many ones
        text = [' '.join(sentence) for sentence in self.tweets["text"]]
        self.tweets["text"] = text

        self.tweets.to_csv("data/clean_tweets.csv", index=True)

    def clean_prices(self):

        #delete zeros columns
        self.prices = self.prices.loc[:, (self.prices != 0).any(axis=0)]
        self.prices["Date"] = pd.to_datetime(self.prices["Date"], format='%Y/%m/%d').dt.date

        # drop duplicate
        self.prices = self.prices.drop_duplicates(subset='Date', keep="last")
        self.prices.to_csv("data/clean_prices.csv", index=False)

    def merge_data(self):
        tweets = pd.read_csv("data/tweets_scores.csv")
        #tweets["date"]=pd.to_datetime(tweets["date"], format='%Y/%m/%d').dt.date
        prices = pd.read_csv("data/clean_prices.csv")

        tweets_dates = list(tweets["date"])
        prices_dates = list(prices["Date"])

        for i in prices_dates:
            if not i in tweets_dates:
                print("insérer ",i," dans tweets")
                tweets = tweets.append({'date': i,
                                        'positive': tweets["positive"].median(),
                                        'negative':tweets['negative'].median()
                                        },
                                       ignore_index=True)
        """
        print("####################################\n\n")

        for i in tweets_dates:
            if not i in prices_dates:
                print("insérer ",i, " dans prices")
                prices = prices.append({'Date': i,
                                        'High': prices["High"].median(),
                                        'Low':prices['Low'].median(),
                                        'Open':prices['Open'].median(),
                                        'Close':prices['Close'].median(),
                                        'Volume':prices['Volume'].median(),
                                        'Adj Close':prices['Adj Close'].median()
                                        },
                                       ignore_index=True
                                       )"""


        tweets=tweets.sort_values("date", ascending=True)
        tweets.to_csv("data/tweets_scores_.csv", index=False)
        prices=prices.sort_values("Date", ascending=True)
        df = tweets.rename(columns={'date': 'Date'})
        df = prices.merge(df, on="Date")
        df.to_csv("data/merged.csv", index=False)

#-----------------------------------------------------------------------------------------------------#

class LSTM_data():
    def __init__(self):
        self.raw_data= pd.read_csv("data/merged.csv",
                                    index_col=0
                                    )[["Close","High","Low","Open","Volume","positive","negative"]]

        self.lag:int = 5
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()


    def X(self):
        return self.raw_data[["High","Low","Open","Volume","positive","negative"]].values

    def y(self):
        return self.raw_data[["Close"]].values
    def add_noise(self):
        mu, sigma = 0, 0.1
        # creating a noise with the same dimension as the dataset (2,2)
        noise = np.random.normal(mu, sigma, self.raw_data[["positive", "negative"]].shape)
        self.raw_data[["positive", "negative"]]= np.add(self.raw_data[["positive", "negative"]].values,noise)



    def lag_func(self,data):
        xs = []
        ys = []

        for i in range(len(data) - self.lag - 1):
            x = data[i:(i + self.lag)]
            y = data[i + self.lag, 0]
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def normalize_x(self, values):
        # normalize features
        return self.x_scaler.fit_transform(values)

    def normalize_y(self, values):
        # normalize features
        return self.y_scaler.fit_transform(values)

    def get_memory(self):
        # add gaussian noise to tweets scores
        self.add_noise()
        #get data
        X, y = self.X(), self.y()

        #X, y = self.normalize_x(X), self.normalize_y(y)
        y = y.reshape(-1, 1)
        X_train, X_test = X[:int(X.shape[0] * .8), :], X[int(X.shape[0] * .8):, :]

        y_train, y_test = y[:int(X.shape[0] * .8), :], y[int(X.shape[0] * .8):, :]

        X_train = self.normalize_x(X_train)
        y_train = self.normalize_y(y_train)
        y_test = y_test.reshape(-1, 1)
        train = np.concatenate((y_train, X_train), axis=1)
        X_train, y_train = self.lag_func(train)

        X_test = self.x_scaler.transform(X_test)
        y_test = self.y_scaler.transform(y_test)
        y_test = y_test.reshape(-1, 1)
        test = np.concatenate((y_test, X_test), axis=1)
        #y_test = y_test.reshape(-1, 1)

        X_test, y_test = self.lag_func(test)

        return X_train, y_train, X_test, y_test

        """print(y.shape)
        # split te data to test and train dataset
        X_train, X_test= X[:int(X.shape[0]*.8), :], X[int(X.shape[0]*.8):,:]
    
        y_train, y_test = y[:int(X.shape[0]*.8), :], y[int(X.shape[0]*.8):,:]
    
        y_train = y_train.reshape(-1, 1)
    
        train = np.concatenate((y_train, X_train), axis=1)
        X_train, y_train = self.lag_func(train)
    
        y_test = y_test.reshape(-1, 1)
        test = np.concatenate((y_test, X_test), axis=1)
        X_test, y_test = self.lag_func(test)
    
        return X_train, y_train, X_test, y_test
        """
    def many_to_one(self):

        return

    def plot(self,column):
        print(self.raw_data[["Close", "High"]])
        """"import plotly.express as px
        df = self.raw_data[[column]]
        df = df.diff(10)
        df = df.set_index(self.raw_data.index.values)
        fig = px.line(df[[column]], x=df.index.values, y=column, title=column)
        fig.show()"""


def model1(n_features, n_steps=7):
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))

    model.add(LSTM(100, activation='relu',return_sequences=True, stateful=True))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    return model
if __name__ == "__main__":

    data = LSTM_data()
    data.plot("Close")
    """X_train, y_train, X_test, y_test = data.get_memory()
    model=model1(X_train.shape[1])
    print(X_test.shape)
    history = model.fit(X_train, y_train, epochs=50,  validation_data=(X_test, y_test), verbose=2,
                        shuffle=False)
"""



