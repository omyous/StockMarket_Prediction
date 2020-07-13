from src.scrapping.tweets import *
from src.scrapping.preprocessing import *
from src.scrapping.stock_prices import *


def update_historical_data():
 # get the lastest tweets
 #t = Tweets()
 #t.update_tweet_hist()
 # get the latest stock prices
 s = Stock_data()
 s.update_stock_hist()
 print("update done")


def update_clean_merge():
  update_historical_data()
  #clean and merged the datasets
  df = Custom_dataset()
  """df.clean_data()
  senti = Sentimend_analysis()
  senti.tweets_analytics()"""
  df.merge_data()
from tensorflow.keras import models
if __name__=="__main__":
 #update_clean_merge()
 regressor = models.load_model("data/weights/dense_no_senti")
 data= LSTM_data()
 X_test = data.X_test
 print(X_test.shape)
 X_test= X_test.reshape(X_test.shape[0], X_test.shape[2])
 print(X_test.shape)
 regressor.predict(X_test[-1:,:-12])

