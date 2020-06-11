import src.scrapping.tweets as tweeter
import src.scrapping.stock_prices as yahoo
import pandas as pd


class Custom_dataset():
    def __init__(self):
        self.tweets_path = "data/google_tweets_.csv"
        self.prices_path = "data/google.csv"
        self.tweets = None
        self.prices = None

    def get_data(self):
        tweeter.Tweets().update_tweet_hist()
        yahoo.Stock_data().update_stock_hist()
        self.tweets = pd.read_csv(self.tweets_path)
        self.prices = pd.read_csv(self.prices_path)


    def clean_data(self):
        self.get_data()
        #print(self.prices.shape)
        #print(self.prices[self.prices.duplicated()])
        #group tweets of the same day since our stock prices ar daily
        #self.clean_tweets()
        self.clean_prices()



    def clean_tweets(self):

        self.tweets["date"] = pd.to_datetime(self.tweets["date"], format='%Y/%m/%d')
        self.tweets["date"] = self.tweets["date"].dt.date
        self.tweets['expand'] = self.tweets.apply(lambda x: ', '.join([x['text']]), axis=1)
        self.tweets = self.tweets.groupby('date')['expand'].apply(list)
        self.tweets = pd.DataFrame(data=self.tweets.values, index=self.tweets.index, columns=["text"])
        self.tweets.to_csv("data/clean_tweets.csv")

    def clean_prices(self):
        #drop duplicate
        self.prices = self.prices.drop_duplicates()
        #delete zeros columns
        self.prices = self.prices.loc[:, (self.prices != 0).any(axis=0)]

        self.prices["Date"] = pd.to_datetime(self.prices["Date"], format='%Y/%m/%d')
        self.prices["Date"] = self.prices["Date"].dt.date


if __name__ == "__main__":
    df = Custom_dataset()
    df.clean_data()
    print(df.prices)

    """"  t = df.tweets
    t["date"] = pd.to_datetime(t["date"], format='%Y/%m/%d')

    t["date"]=t["date"].dt.date
    t['expand'] = t.apply(lambda x: ', '.join([x['text']]), axis=1)
    t=t.groupby('date')['expand'].apply(list)
    t = pd.DataFrame(data=t.values, index=t.index, columns=["text"])
    print(t.head())
    """



