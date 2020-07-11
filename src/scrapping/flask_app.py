
from flask import Flask, request, jsonify, render_template
from tensorflow.keras import models
from src.scrapping.preprocessing import *
from src.scrapping.many_to_one import *


app = app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

"""attn_model_no_senti = models.load_model("data/weights/attn_based_lstm_no_senti")
free_attn_model = models.load_model("data/weights/free_attn_lstm")
free_attn_model_no_senti = models.load_model("data/weights/free_attn_lstm_no_senti")
dense_net = models.load_model("data/weights/dense_net")
dense_net = models.load_model("data/weights/dense_net")"""


@app.after_request
def add_header(response):
    return response
@app.route("/")
def home():
    return render_template("index.html", message_bienvenue="Bienvenue sur la page d'accueil !")

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    """data: LSTM_data = LSTM_data()
    preds = attn_model.predict(data.X_test[-1:,-1:,:])

    preds = data.y_scaler.inverse_transform(preds)
    preds = np.round(preds,decimals=2)
    preds=preds[:, 0][0]"""
    preds = .0
    period = request.form['period']
    mode= request.form['mode']
    model = request.form['model']
    preds = load_and_evaluate(period=period, mode=mode, model=model)

    return render_template('predictions.html', prediction_text="Coogle's closure for today: \n{}$".format(preds))


def get_evaluation_dat():
    data = pd.read_csv("data/merged.csv").tail(6)
    Y = data[["Close"]]
    Y.columns = ["Y"]
    # shift the features
    cols = ["High", "Low", "Open", "Volume", "Adj Close", "positive", "negative"]
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
    X.columns = ['High_lag1', 'Low_lag1', 'Open_lag1', 'Volume_lag1', 'Adj Close_lag1',
                 'High_lag2', 'Low_lag2', 'Open_lag2', 'Volume_lag2', 'Adj Close_lag2',
                 'High_lag3', 'Low_lag3', 'Open_lag3', 'Volume_lag3', 'Adj Close_lag3',
                 'High_lag4', 'Low_lag4', 'Open_lag4', 'Volume_lag4', 'Adj Close_lag4',
                 'High_lag5', 'Low_lag5', 'Open_lag5', 'Volume_lag5', 'Adj Close_lag5',
                 'Y_lag1', 'Y_lag2', 'Y_lag3', 'Y_lag4', 'Close',
                 'High', 'Low', 'Open','Volume', 'Adj Close',
                 'positive', 'negative',
                 'positive_lag1', 'negative_lag1',
                 'positive_lag2', 'negative_lag2',
                 'positive_lag3', 'negative_lag3',
                 'positive_lag4', 'negative_lag4',
                 'positive_lag5', 'negative_lag5'
                 ]
    X[["Close"]].values
    return X
def load_and_evaluate(period, mode, model):
    data = LSTM_data()
    if model == "Attention based Lstm model":
        if mode == "Financial sentiment analysis forecasting":
            if period == "Today's closure price":
                regressor = models.load_model("data/weights/attn_based_lstm")
                preds = regressor.predict(data.X_test[-1:,-1:,:])
                preds = data.y_scaler.inverse_transform(preds)
                preds = np.round(preds, decimals=2)[:, 0][0]
        else:
            if period == "Today's closure price":
                regressor = models.load_model("data/weights/attn_based_lstm_no_senti")
                preds = regressor.predict(data.X_test[-1:,-1:,:-2])
                preds = data.y_scaler.inverse_transform(preds)
                preds = np.round(preds, decimals=2)[:, 0][0]

    elif model =="Free attention lstm model":
        if mode == "Financial sentiment analysis forecasting":
            if period == "Today's closure price":
                regressor = models.load_model("data/weights/free_attn_lstm")
                preds = regressor.predict(data.X_test[-1:, -1:, :])
                preds = data.y_scaler.inverse_transform(preds)
                preds = np.round(preds, decimals=2)[:, 0][0]

        else:
            if period == "Today's closure price":
                regressor = models.load_model("data/weights/free_attn_lstm_no_senti")
                preds = regressor.predict(data.X_test[-1:, -1:, :-2])
                preds = data.y_scaler.inverse_transform(preds)
                preds = np.round(preds, decimals=2)[:, 0][0]

    else:
        if mode == "Financial sentiment analysis forecasting":
            if period == "Today's closure price":
                regressor = models.load_model("data/weights/dense_net")
                preds = regressor.predict(data.X_test[-1, :])
                preds = data.y_scaler.inverse_transform(preds)
                preds = np.round(preds, decimals=2)[:, 0][0]
        else:
            if period == "Today's closure price":
                    regressor = models.load_model("data/weights/dense_net")
                    preds = regressor.predict(data.X_test[-1, :-2])
                    preds = data.y_scaler.inverse_transform(preds)
                    preds = np.round(preds, decimals=2)[:, 0][0]


    return preds



@app.route('/predictions.html')
def predictions():
    return render_template('predictions.html')

@app.route('/news.html')
def actualites():
    return render_template('news.html')
@app.route("/contact.html")
def news():
    return render_template('contact.html')


if __name__ == "__main__":

    app.run(debug=True)