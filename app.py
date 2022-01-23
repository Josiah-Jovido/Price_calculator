#Import dependencies
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
#import xgboost as xgb
import sys
from joblib import dump, load
import pickle

app = Flask(__name__)
""" load model and assign to variable """

rf_model = load('rf_model.pkl') 
print(' RF Model Loaded')
rf = rf_model["rf_model"]
ada_boost = load('ada_boost.pkl')
print('Ada boost model loaded')
ada = ada_boost["ada_boost"]
DT_model = load('DT_model.pkl')
print('DT model loaded')
dt = DT_model["DT_model"]
lin_model = load('lin_model.pkl')
print('Lin model loaded')
lin = lin_model["lin_model"]
ridge_model = load('ridge_model.pkl')
print('Ridge model loaded')
ridge = ridge_model["ridge_model"]
svr = load('svr.pkl')
print('SVR model loaded')
svr_model = svr["svr"]

y_test = np.array([200000])

@app.route('/')
def home():
    """ Render homepage """
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,-1)
    prediction1 = rf.predict(final_features)
    prediction2 = ada.predict(final_features)
    prediction3 = dt.predict(final_features)
    prediction4 = lin.predict(final_features)
    prediction5 = ridge.predict(final_features)
    prediction6 = svr_model.predict(final_features)

    # R squared value
    r_sqd1 = metrics.mean_absolute_error(y_test, prediction1)
    r_sqd2 = metrics.mean_absolute_error(y_test, prediction2)
    r_sqd3 = metrics.mean_absolute_error(y_test, prediction3)
    r_sqd4 = metrics.mean_absolute_error(y_test, prediction4)
    r_sqd5 = metrics.mean_absolute_error(y_test, prediction5)
    r_sqd6 = metrics.mean_absolute_error(y_test, prediction6)

    return render_template('result.html', pred_text1='The estimated cost of house is {}'.format(prediction1),
    pred_text2='The estimated cost of house is {}'.format(prediction2),
    pred_text3='The estimated cost of house is {}'.format(prediction3),
    pred_text4='The estimated cost of house is {}'.format(prediction4),
    pred_text5='The estimated cost of house is {}'.format(prediction5),
    pred_text6='The estimated cost of house is {}'.format(prediction6),
    sq1 = 'The Mean Absolute Error value is {}'.format(r_sqd1),
    sq2 = 'The Mean Absolute Error value is {}'.format(r_sqd2),
    sq3 = 'The Mean Absolute Error value is {}'.format(r_sqd3),
    sq4 = 'The Mean Absolute Error value is {}'.format(r_sqd4),
    sq5 = 'The Mean Absolute Error value is {}'.format(r_sqd5),
    sq6 = 'The Mean Absolute Error value is {}'.format(r_sqd6)
    )

if __name__ == "__main__":
    try:
        port = int(sys.argv[1]) # incase a command line port argument is specified use it as default port
    except:
        port = 5200 # if not use this
    print(sys.argv)
    app.run(port=port, debug=True)
