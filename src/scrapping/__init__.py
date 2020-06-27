from src.scrapping.tweets import *
from src.scrapping.preprocessing import *
from src.scrapping.stock_prices import *


def update_historical_data():
 # get the lastest tweets
 t = Tweets()
 t.update_tweet_hist()
 # get the latest stock prices
 s = Stock_data()
 s.update_stock_hist()


def clean_and_merge():
 #update_historical_data()
 #clean and merged the datasets
 df = Custom_dataset()
 df.clean_data()
 senti = Sentimend_analysis()
 senti.tweets_analytics()
 df.merge_data()

if __name__=="__main__":
 clean_and_merge()

