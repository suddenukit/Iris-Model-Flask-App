# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:06:47 2022

@author: Haoyun Chen
"""

import os
import numpy as np
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from iris_model import train_model
from sklearn.externals import joblib
import unicodedata
import json

app = Flask(__name__)
api = Api(app)
if not os.path.isfile('iris_model.model'):
    train_model()
model = joblib.load('iris_model.model')

class MakePrediction(Resource):
    @staticmethod
    def post():
        # acquire inputs
        posted_data = request.data
        posted_data = json.loads(posted_data)
        sepal_length=posted_data.get("sepal_length")
        sepal_width=posted_data.get("sepal_width")
        petal_length=posted_data.get("petal_length")
        petal_width=posted_data.get("petal_width")  
        # input process
        if sepal_length is None:
            return jsonify({"code":0,"msg":"sepal length is empty"})
        if sepal_width is None:
            return jsonify({"code":0,"msg":"sepal width is empty"})
        if petal_length is None:
            return jsonify({"code":0,"msg":"petal length is empty"})
        if petal_width is None:
            return jsonify({"code":0,"msg":"petal width is empty"})
        if not is_number(sepal_length):
            return jsonify({"code":0,"msg":"sepal length  is not a valid number."})
        if not is_number(sepal_width):
            return jsonify({"code":0,"msg":"sepal width  is not a valid number."})
        if not is_number(petal_length):
            return jsonify({"code":0,"msg":"petal length  is not a valid number."})
        if not is_number(petal_width):
            return jsonify({"code":0,"msg":"petal width  is not a valid number."})
        # Make prediction
        X_test = np.array([[sepal_length, sepal_width, petal_length, petal_width]])  
        X_test = X_test.astype(float)
        prediction = model.predict(X_test)[0]
        print("The prediction: ",prediction)
        if prediction == '0':
            return jsonify({'Prediction': 'setosa'})
        elif prediction == '1':
            return jsonify({'Prediction': 'versicolor'})
        elif prediction == '2':
            return jsonify({'Prediction': 'virginica'})
        return
api.add_resource(MakePrediction, '/predict')

# Judge whether a input is a valid float
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError,ValueError):
        pass
    return False

if __name__ == '__main__':
    app.run(port=5000)