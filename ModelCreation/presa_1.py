from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import MinMaxScaler


import tensorflow as tf


import joblib




dataset = read_csv("C:/Users/Arturo A/AguayDrenaje/dbCSV/Presa1.csv", parse_dates=['Date'], index_col='Date')

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
scaler_filename = "scalerPresa1.save"
joblib.dump(scaler, scaler_filename) 
# And now to load to check it worked fine
scaler = joblib.load(scaler_filename) 


# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[3]], axis=1, inplace=True)

# split into train and test sets
values = reframed.values
n_train_days = 365*12
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
model.fit(X_train, y_train, epochs=50, batch_size=True, validation_data=(X_test, y_test), verbose=True, shuffle=False)

model.save('C:\\Users\\Arturo A\\AguayDrenaje\\mdlPresa1')




