import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras import models
app = app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#model = models.load_model("data/weights/attn_based_lstm.h5")

"""# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), model.predict(test_input)
)
"""
# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:

@app.after_request
def add_header(response):
    return response

@app.route("/")
def home():
    return render_template("index.html", message_bienvenue="Bienvenue sur la page d'accueil !")

@app.route('/predictions.html')
def predictions():
    return render_template('predictions.html')

@app.route('/actualities.html')
def actualites():
    return render_template('actualities.html')


if __name__ == "__main__":

    app.run(debug=True)