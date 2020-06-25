import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('../machine_learning/model.pkl', 'rb'))
@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictions.html')
def predictions():
    return render_template('predictions.html')

@app.route('/actualities.html')
def actualites():
    return render_template('actualities.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The prices will propably be as follow $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True)