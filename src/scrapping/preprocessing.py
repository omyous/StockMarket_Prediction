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


if __name__ == "__main__":
    df = Custom_dataset()
    df.get_data()
    print(df.tweets.head())


