from flask import Flask, render_template, request, url_for, Markup, jsonify, redirect, flash, send_file, send_from_directory
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import utils
import train_models as tm
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)

def init_models():
    try:
        pol = pickle.load(open('poly.pkl', 'rb'))
        regresso = pickle.load(open('regressor.pkl', 'rb'))
    except (FileNotFoundError, ModuleNotFoundError, ImportError):
        pol = PolynomialFeatures(degree=2)
        regresso = LinearRegression()
        pickle.dump(pol, open('poly.pkl', 'wb'))
        pickle.dump(regresso, open('regressor.pkl', 'wb'))
    return pol, regresso

pol, regresso = init_models()

def perform_training(stock_name, df, models_list):
    all_colors = {
        'SVR_linear': '#FF9EDD',
        'SVR_poly': '#FFFD7F',
        'svr_rbf': '#FFA646',
        'linear_regression': '#CC2A1E',
        'random_forests': '#8F0099',
        'knn': '#CCAB43',
        'elastic_net': '#CFAC43',
        'dt': '#85CC43',
        'lstm_model': '#CC7674',
        'rnn_model': '#CC7687'
    }

    print(df.head())
    dates, prices, ml_models_outputs, prediction_date, test_price = tm.train_predict_plot(stock_name, df, models_list)
    origdates = dates
    if len(dates) > 20:
        dates = dates[-20:]
        prices = prices[-20:]

    all_data = []
    all_data.append((prices, 'false', 'Data', '#000000'))

    for model_output, model_data in ml_models_outputs.items():
        if model_data is None:
            print(f"[WARNING] Skipping model with None output: {model_output}")
            continue
        try:
            output_data = model_data[0]
            if len(origdates) > 20:
                output_data = output_data[-20:]
            all_data.append((output_data, "true", model_output, all_colors.get(model_output, "#333333")))
        except Exception as e:
            print(f"[ERROR] Failed to process model {model_output}: {e}")
            continue

    all_prediction_data = [("Original", test_price)]
    all_test_evaluations = []

    for model_output, model_data in ml_models_outputs.items():
        if model_data is None:
            continue
        try:
            all_prediction_data.append((model_output, model_data[1]))
            all_test_evaluations.append((model_output, model_data[2]))
        except Exception as e:
            print(f"[ERROR] Prediction/Test Eval failed for model {model_output}: {e}")
            continue

    return all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations

all_files = utils.read_all_stock_files('individual_stocks_5yr')

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')    

@app.route('/abstract')
def abstract():
    return render_template('abstract.html')

@app.route('/future')
def future():
    return render_template('future.html')    

@app.route('/login')
def login():
    return render_template('login.html')
    
@app.route('/upload')
def upload():
    return render_template('upload.html')  

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df) 

@app.route('/landing_function')
def landing_function():
    stock_files = list(all_files.keys())
    return render_template('index.html', show_results="false", stocklen=len(stock_files), stock_files=stock_files, len2=len([]),
                           all_prediction_data=[], prediction_date="", dates=[], all_data=[], len=len([]))

@app.route('/process', methods=['POST'])
def process():
    stock_file_name = request.form['stockfile']
    ml_algorithms = request.form.getlist('mlalgos')
    df = all_files[str(stock_file_name)]

    all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations = perform_training(
        str(stock_file_name), df, ml_algorithms)

    stock_files = list(all_files.keys())

    return render_template('index.html', all_test_evaluations=all_test_evaluations, show_results="true",
                           stocklen=len(stock_files), stock_files=stock_files, len2=len(all_prediction_data),
                           all_prediction_data=all_prediction_data, prediction_date=prediction_date,
                           dates=dates, all_data=all_data, len=len(all_data))

@app.route('/prediction')
def prediction():
    return render_template("prediction.html")

@app.route('/predict', methods=['POST'])
def predict():
    int_feature = [x for x in request.form.values()]
    final_features = [np.array(int_feature)]
    Total_infections = pol.transform(final_features)
    prediction = regresso.predict(Total_infections)
    pred = format(int(prediction[0]))
    return render_template('prediction.html', prediction_text=pred)    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
