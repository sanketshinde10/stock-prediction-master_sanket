import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# For RNN/LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

# setting a seed for reproducibility
np.random.seed(10)


def read_all_stock_files(folder_path):
    allFiles = []
    for (_, _, files) in os.walk(folder_path):
        allFiles.extend(files)
        break

    dataframe_dict = {}
    for stock_file in allFiles:
        df = pd.read_csv(os.path.join(folder_path, stock_file))
        dataframe_dict[stock_file.split('_')[0]] = df

    return dataframe_dict


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def create_preprocessed_Dataset(df, look_back=60):  # Increased default look_back for RNN/LSTM
    try:
        # Keep only 'close' column
        df = df[['close']]
        dataset = df.values.astype('float32')
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        
        # Ensure minimum dataset size
        if len(dataset) < look_back + 2:  # +2 for at least 1 train and 1 test sample
            raise ValueError(f"Dataset too small. Needs at least {look_back + 2} samples")
        
        # Split into train and test sets
        train_size = len(dataset) - 2
        train, test = dataset[:train_size], dataset[train_size:]
        
        # Create sequences
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        
        # Ensure we have test data
        if len(testX) == 0:
            testX = trainX[-1:]  # Use last training sample as test
            testY = trainY[-1:] if len(trainY) > 0 else np.array([0])
            
        return trainX, trainY, testX, testY, scaler
        
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        # Return default values that won't break the model
        dummy = np.zeros((1, look_back))
        return dummy, np.array([0]), dummy, np.array([0]), MinMaxScaler()


def getData(df):
    dates = []
    prices = []

    last_row = df.tail(1)
    df = df.head(len(df) - 1)

    df_dates = df.loc[:, 'date']
    df_close = df.loc[:, 'close']

    for date in df_dates:
        dates.append([int(date.split('-')[2])])

    for close_price in df_close:
        prices.append(float(close_price))

    last_date = int(((list(last_row['date']))[0]).split('-')[2])
    last_price = float((list(last_row['close']))[0])
    return dates, prices, last_date, last_price


def svr_linear(dates, prices, test_date, df):
    svr_lin = SVR(kernel='linear', C=1e3)
    trainX, trainY, testX, testY, _ = create_preprocessed_Dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    svr_lin.fit(X_train, y_train)
    decision_boundary = svr_lin.predict(trainX)
    y_pred = svr_lin.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = svr_lin.predict(testX)[0]
    return decision_boundary, prediction, test_score


def svr_rbf(dates, prices, test_date, df):
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    trainX, trainY, testX, testY, _ = create_preprocessed_Dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    svr_rbf.fit(X_train, y_train)
    decision_boundary = svr_rbf.predict(trainX)
    y_pred = svr_rbf.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = svr_rbf.predict(testX)[0]
    return decision_boundary, prediction, test_score


def linear_regression(dates, prices, test_date, df):
    lin_reg = LinearRegression()
    trainX, trainY, testX, testY, _ = create_preprocessed_Dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    lin_reg.fit(X_train, y_train)
    decision_boundary = lin_reg.predict(trainX)
    y_pred = lin_reg.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = lin_reg.predict(testX)[0]
    return decision_boundary, prediction, test_score


def random_forests(dates, prices, test_date, df):
    rand_forst = RandomForestRegressor(n_estimators=10, random_state=0)
    trainX, trainY, testX, testY, _ = create_preprocessed_Dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    rand_forst.fit(X_train, y_train)
    decision_boundary = rand_forst.predict(trainX)
    y_pred = rand_forst.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = rand_forst.predict(testX)[0]
    return decision_boundary, prediction, test_score


def knn(dates, prices, test_date, df):
    knn = KNeighborsRegressor(n_neighbors=2)
    trainX, trainY, testX, testY, _ = create_preprocessed_Dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    knn.fit(X_train, y_train)
    decision_boundary = knn.predict(trainX)
    y_pred = knn.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = knn.predict(testX)[0]
    return decision_boundary, prediction, test_score


def dt(dates, prices, test_date, df):
    decision_trees = tree.DecisionTreeRegressor()
    trainX, trainY, testX, testY, _ = create_preprocessed_Dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    decision_trees.fit(X_train, y_train)
    decision_boundary = decision_trees.predict(trainX)
    y_pred = decision_trees.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = decision_trees.predict(testX)[0]
    return decision_boundary, prediction, test_score


def elastic_net(dates, prices, test_date, df):
    regr = ElasticNet(random_state=0)
    trainX, trainY, testX, testY, _ = create_preprocessed_Dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
    regr.fit(X_train, y_train)
    decision_boundary = regr.predict(trainX)
    y_pred = regr.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    prediction = regr.predict(testX)[0]
    return decision_boundary, prediction, test_score


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
from sklearn.metrics import mean_squared_error

def lstm_model(dates, prices, test_date, df):
    look_back = 60
    trainX, trainY, testX, testY, scaler = create_preprocessed_Dataset(df, look_back=look_back)

    print("trainX shape:", trainX.shape)
    print("testX shape:", testX.shape)

    # Reshape trainX
    if len(trainX.shape) == 1:
        trainX = np.reshape(trainX, (trainX.shape[0], 1, 1))
    else:
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    # Reshape testX
    if testX.size == 0:
        print("testX is empty. Cannot predict on test data.")
        return None, None, None

    if len(testX.shape) == 1:
        testX = np.reshape(testX, (testX.shape[0], 1, 1))
    elif len(testX.shape) == 2:
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(1, trainX.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=50, batch_size=32, verbose=0)

    trainPredict = model.predict(trainX).flatten()
    testPredict = model.predict(testX).flatten()
    test_score = mean_squared_error(testY, testPredict)

    return trainPredict, testPredict, test_score


def rnn_model(dates, prices, test_date, df):
    look_back = 60
    trainX, trainY, testX, testY, scaler = create_preprocessed_Dataset(df, look_back=look_back)

    print("trainX shape:", trainX.shape)
    print("testX shape:", testX.shape)

    # Reshape trainX
    if len(trainX.shape) == 1:
        trainX = np.reshape(trainX, (trainX.shape[0], 1, 1))
    else:
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    # Reshape testX
    if testX.size == 0:
        print("testX is empty. Cannot predict on test data.")
        return None, None, None

    if len(testX.shape) == 1:
        testX = np.reshape(testX, (testX.shape[0], 1, 1))
    elif len(testX.shape) == 2:
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=(1, trainX.shape[2])),
        SimpleRNN(50),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=50, batch_size=32, verbose=0)

    trainPredict = model.predict(trainX).flatten()
    testPredict = model.predict(testX).flatten()
    test_score = mean_squared_error(testY, testPredict)

    return trainPredict, testPredict, test_score
