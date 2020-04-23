# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:53:48 2020

@author: ArturoA
"""

from math import sqrt
from numpy import concatenate
from datetime import datetime
import numpy as np


import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

import tensorflow as tf

dataset = read_csv("C:/Users/Arturo A/AguayDrenaje/dbCSV/postQData2.csv", parse_dates=['date'], index_col='date')

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

values = dataset.values

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit(values)
scaled = scaler.transform(values)

#Send export scalar to be used by flask later
scaler_filename = "scalerLSTM.save"
joblib.dump(scaler, scaler_filename) 

# And now to load to check it worked fine

scaler = joblib.load(scaler_filename) 


# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]], axis=1, inplace=True)

# split into train and test sets
values = reframed.values
n_train_days = 365*5
train = values[:n_train_days, :]
test = values[n_train_days:, :]

# split into input and outputs
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# design network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(70, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
model.fit(X_train, y_train, epochs=50, batch_size=True, validation_data=(X_test, y_test), verbose=True, shuffle=False)

model.save('C:\\Users\\Arturo A\\AguayDrenaje\\LSTM')

new_model2 = tf.models.load('C:\\Users\\Arturo A\\AguayDrenaje\\LSTM')

# Recreate the exact same model purely from the file
new_model = new_model = tf.keras.models.load_model('testDiario')

prediction = new_model.predict(X_test)

# make a prediction
y_pred = model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))

print(y_pred)