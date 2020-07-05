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
 df.clean_data()
 senti = Sentimend_analysis()
 senti.tweets_analytics()
 df.merge_data()
from tensorflow.keras import models
from src.scrapping.many_to_one import *
from src.scrapping.attention import *
if __name__=="__main__":
 #update_clean_merge()

  X_train, X_test, Y_train, Y_test = LSTM_data().get_memory()
  # X_train, X_test = X_train[:, :, :-2], X_test[:, :, :-2]

  i = Input(shape=(X_train.shape[1], X_train.shape[2]))

  att_in = LSTM(NEURONS,
                return_sequences=True,
                activation=ACTIVATION,
                activity_regularizer=regularizers.l2(L2),
                bias_regularizer=regularizers.l2(BIAIS_REG),
                )(i)
  att_in = LSTM(NEURONS,
                return_sequences=True,
                activation=ACTIVATION,
                activity_regularizer=regularizers.l2(L2),
                bias_regularizer=regularizers.l2(BIAIS_REG),
                )(att_in)
  att_in = LSTM(NEURONS,
                return_sequences=True,
                activation=ACTIVATION,
                activity_regularizer=regularizers.l2(L2),
                bias_regularizer=regularizers.l2(BIAIS_REG),
                )(att_in)
  att_out = attention()(att_in)
  att_out = Dropout(DROPOUT)(att_out)
  outputs = Dense(1,
                  activation='relu',
                  trainable=True,
                  bias_regularizer=regularizers.l2(BIAIS_REG),
                  activity_regularizer=regularizers.l2(L2)
                  )(att_out)

  model = Model(inputs=[i], outputs=[outputs])
  load = models.load_model("data/weights/attn_based_lstm")
  print(load)

