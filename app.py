import pickle

from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np 
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("linear_regression.pkl", 'rb'))
scalar = pickle.load(open("scaling.pkl",'rb')) 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    transformed_data = scalar.transform(np.array(list(data.values().reshape(1,-1))))
    pred = model.predict(transformed_data)
    print(pred[0])
    return jsonify(pred[0])

@app.route('/predict',methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    scaled_data = scalar.transform(np.array(data).reshape(1,-1))
    print(scaled_data)
    pred = model.predict(scaled_data)
    return render_template("home.html", prediction_text = "The House price prediction is {}".format(pred[0]))


if __name__ == "__main__":
    app.run(debug=True)