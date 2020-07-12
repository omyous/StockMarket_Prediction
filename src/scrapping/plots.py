from matplotlib import pyplot
import pandas as pd
from sklearn.utils import check_array
import numpy as np
def mean_absolute_percentage_error(y_true, y_pred):
    #y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def mean_percentage_error(y_true, y_pred):
    #y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean((y_true - y_pred) / y_true) * 100
def attn_performances():
    df = pd.read_csv("data/attn_evaluate_senti_yes.csv", index_col=2)
    df_no = pd.read_csv("data/attn_evaluate_senti_no.csv", index_col=2)

    y = df["Close"].values
    # y = y.reshape(-1,1)
    y_preds = df["predictions"].values
    # y_preds = y_preds.reshape(-1, 1)
    print("MAPE", mean_absolute_percentage_error(y, y_preds))
    print("MPE", mean_percentage_error(y, y_preds))
    from sklearn.metrics import mean_squared_error
    print("MSE", mean_squared_error(y, y_preds))
    from sklearn.metrics import accuracy_score
    print("Accuracy", accuracy_score(y.round(), y_preds.round()))

    pyplot.figure(figsize=(15, 7))
    pyplot.plot(df["Close"])
    pyplot.plot(df['predictions'])
    pyplot.plot(df_no['predictions'])
    pyplot.title('attn_based_LSTM with sentiments analysis vs attn_based_LSTM without sentiments analysis')
    pyplot.ylabel('Close values')
    pyplot.xlabel('dates')
    pyplot.legend(['real Close', 'attn_senti_Close', "atnn_no_senti_Close"], loc='upper left')
    degrees = 70
    pyplot.xticks(rotation=degrees)
    pyplot.show()
def free_attn_performances():
    df = pd.read_csv("data/free_attn_lstm_senti_yes.csv", index_col=2)
    df_no = pd.read_csv("data/free_attn_lstm_senti_no.csv", index_col=2)

    y = df["Close"].values
    # y = y.reshape(-1,1)
    y_preds = df["predictions"].values
    # y_preds = y_preds.reshape(-1, 1)
    print("MAPE", mean_absolute_percentage_error(y, y_preds))
    print("MPE", mean_percentage_error(y, y_preds))
    from sklearn.metrics import mean_squared_error
    print("MSE", mean_squared_error(y, y_preds))
    from sklearn.metrics import accuracy_score
    print("Accuracy", accuracy_score(y.round(), y_preds.round()))

    pyplot.figure(figsize=(15, 7))
    pyplot.plot(df["Close"])
    pyplot.plot(df['predictions'])
    pyplot.plot(df_no['predictions'])
    pyplot.title('free_attn_LSTM with sentiment_analysis vs free_attn_LSTM without sentiment analysis')
    pyplot.ylabel('Close values')
    pyplot.xlabel('dates')
    pyplot.legend(['real Close', 'free_attn_LSTM_senti_Close', "free_atnn_LSTM_no_senti_Close"], loc='upper left')
    degrees = 70
    pyplot.xticks(rotation=degrees)
    pyplot.show()

def denseNet_performances():
    df = pd.read_csv("data/dense_senti_yes.csv", index_col=2)
    df_no = pd.read_csv("data/dense_senti_no.csv", index_col=2)

    y = df["Close"].values
    # y = y.reshape(-1,1)
    y_preds = df["predictions"].values
    # y_preds = y_preds.reshape(-1, 1)
    print("MAPE", mean_absolute_percentage_error(y, y_preds))
    print("MPE", mean_percentage_error(y, y_preds))
    from sklearn.metrics import mean_squared_error
    print("MSE", mean_squared_error(y, y_preds))
    from sklearn.metrics import accuracy_score
    print("Accuracy", accuracy_score(y.round(), y_preds.round()))

    pyplot.figure(figsize=(15, 7))
    pyplot.plot(df["Close"])
    pyplot.plot(df['predictions'])
    pyplot.plot(df_no['predictions'])
    pyplot.title('denseNet with sentiment_analysis vs denseNet without sentiment analysis')
    pyplot.ylabel('Close values')
    pyplot.xlabel('dates')
    pyplot.legend(['real Close', 'denseNet_senti_Close', "denseNet_no_senti_Close"], loc='upper left')
    degrees = 70
    pyplot.xticks(rotation=degrees)
    pyplot.show()

def plot_times_serie():
    pyplot.figure(figsize=(10, 7))
    df = pd.read_csv("data/merged.csv",  index_col=0).tail(40)
    pyplot.plot(df["Close"])
    degrees = 70
    pyplot.xticks(rotation=degrees)
    pyplot.title("Google's stock close from 13-05-2020 to 09-07-2020")
    pyplot.xlabel('Dates')
    pyplot.ylabel('Close')
    pyplot.grid()
    pyplot.show()
if __name__=="__main__":
    #free_attn_performances()
    #denseNet_performances()
    #attn_performances()
    plot_times_serie()