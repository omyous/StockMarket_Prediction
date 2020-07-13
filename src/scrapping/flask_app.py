
from flask import Flask, request, jsonify, render_template
from tensorflow.keras import models
from src.scrapping.models import *


app = app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def add_header(response):
    return response
#redirect the running application to the home page index.html
@app.route("/")
def home():
    return render_template("index.html", message_bienvenue="Bienvenue sur la page d'accueil !")

#get the fiel
@app.route('/predict',methods=['POST', 'GET'])
def predict():

    preds = .0
    mode= request.form['mode']
    model = request.form['model']
    preds = load_and_evaluate( mode=mode, model=model)

    return render_template('predictions.html', prediction_text="Coogle's closure price for today would be: \n{}$".format(preds))


def load_and_evaluate( mode, model):
    data = LSTM_data()
    if model == "Attention based Lstm model":
        if mode == "Financial sentiment analysis forecasting":
                regressor = models.load_model("data/weights/attn_based_lstm_senti")
                preds = regressor.predict(data.X_test[-1:,-1:,:])
                preds = data.y_scaler.inverse_transform(preds)
                preds = np.round(preds, decimals=2)[:, 0][0]
        else:
                regressor = models.load_model("data/weights/attn_based_lstm_no_senti")
                preds = regressor.predict(data.X_test[-1:,-1:,:-12])
                preds = data.y_scaler.inverse_transform(preds)
                preds = np.round(preds, decimals=2)[:, 0][0]

    elif model =="Free attention lstm model":
        if mode == "Financial sentiment analysis forecasting":
                regressor = models.load_model("data/weights/free_attn_lstm_senti")
                preds = regressor.predict(data.X_test[-1:, -1:, :])
                preds = data.y_scaler.inverse_transform(preds)
                preds = np.round(preds, decimals=2)[:, 0][0]

        else:
                regressor = models.load_model("data/weights/free_attn_lstm_no_senti")
                preds = regressor.predict(data.X_test[-1:, -1:, :-12])
                preds = data.y_scaler.inverse_transform(preds)
                preds = np.round(preds, decimals=2)[:, 0][0]

    else:
        X_test = data.X_test
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])
        if mode == "Financial sentiment analysis forecasting":
                regressor = models.load_model("data/weights/dense_senti")
                preds = regressor.predict(X_test[-1:, :])
                preds = data.y_scaler.inverse_transform(preds)
                preds = np.round(preds, decimals=2)[:, 0][0]
        else:
                    regressor = models.load_model("data/weights/dense_no_senti")
                    preds = regressor.predict(X_test[-1:, :-12])
                    print("prediction done")
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