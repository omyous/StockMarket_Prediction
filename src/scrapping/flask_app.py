
from flask import Flask, request, jsonify, render_template
from tensorflow.keras import models
from src.scrapping.preprocessing import *
from src.scrapping.many_to_one import *

app = app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def add_header(response):
    return response

@app.route("/")
def home():
    return render_template("index.html", message_bienvenue="Bienvenue sur la page d'accueil !")

@app.route('/predict',methods=['POST'])
def predict():

    model = models.load_model("data/weights/attn_based_lstm")
    data = LSTM_data()
    preds = model.predict(data.X_test[-1:,-1:,:])

    preds = data.y_scaler.inverse_transform(preds)
    preds = np.round(preds,decimals=2)
    preds=preds[:, 0][0]

    return render_template('predictions.html', prediction_text='The Google closure price for tomorrow should be {} $'.format(preds))


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