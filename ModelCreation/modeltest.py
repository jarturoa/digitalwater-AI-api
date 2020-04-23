# Importing the libraries
from numpy import loadtxt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
import numpy as np

#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import mean_squared_error
#from numpy import concatenate
#import keras
#from tensorflow.keras.models import load_model
#from math import sqrt
#from matplotlib import pyplot
#from datetime import datetime
#from pandas import read_csv
#from pandas import DataFrame
#from pandas import concat



# load the dataset

ds = loadtxt('C:/Users/Arturo A/AguayDrenaje/dbCSV/MM8.csv', delimiter=',')
ds2 = loadtxt('C:/Users/Arturo A/AguayDrenaje/dbCSV/MM9.csv', delimiter=',')
#normalize
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(ds)

ds2 = scaler.transform(ds2)
XAle = ds2[:,0:17]
# put features and prediction in separate parts
# split into input (X) and output (y) variables
X = scaled[:,0:17]
y = scaled[:,17]
#divide dataset train validation test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# define the keras model
#from keras.layers import LSTM
#stadard
model = Sequential()
#input 8 first hidden layer of 12
model.add(Dense(12, input_dim=17, activation='relu'))
#goes second hidden layer with 8 
model.add(Dense(24, activation='relu'))
#final output layer
model.add(Dense(1, activation='relu'))
# compile the keras model
#just a test to change the learning rate
#we tried with SDG, it didn't work
model.compile(loss='mean_squared_error', optimizer= 'adam', metrics=['mse', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
# fit the keras model on the dataset
#use first one for final model for production
#model.fit(X_train, y_train, epochs=50, batch_size=12)
model.fit(X_train, y_train, epochs=2000, batch_size=12)
#score
score = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
#make a prediction
prediction1 = model.predict(XAle)

print(prediction1)


predplusnormalized = np.column_stack([XAle,prediction1[:,-1]])
inverted = scaler.inverse_transform(predplusnormalized)
result = inverted[:,17]
print(result)

# RICKY TODO Save the model
import tensorflow as tf
model.save('C:\\Users\\Arturo A\\AguayDrenaje\\modeloDiario')
#RICKY Send export scalar to be used by flask later
from sklearn.externals import joblib
scaler_filename = "C:\\Users\\Arturo A\\AguayDrenaje\\testScaler"
joblib.dump(scaler, scaler_filename) 
#YOU DON;T REALLY NEED SCALED
scaled_filename = "C:\\Users\\Arturo A\\AguayDrenaje\\testScaled"
joblib.dump(scaled, scaled_filename)
#RICKY 
# Recreate the exact same model purely from the file
new_model = new_model = tf.keras.models.load_model('C:\\Users\\Arturo A\\AguayDrenaje\\modeloDiario')
#RICKY test if model is imported correctly making a test
prediction = new_model.predict(X_test)
print(prediction)

#importar el scaler para probar que si lo exporte correctamente
#scaler = Path("testScaler")
scaler_filename = scaler
importedScalar = joblib.load(scaler_filename)



#Stuff to do with the API in heroku
from numpy import array

zero = array([[0]])
X_test1 = array([[0.0,0.6,0.506944,0.6,0.055236,0.374126,0.724907,0.199052,0.786765,0.302954,0.261838,0.031014,0.35467,0.223484,0.405913,1.0,0.034749]])
X_text2 = array([[1,4,4289948.83776,229.5,84.39582472,101.50930605,461.955377,942.071072,268.45742538,831.760486,5.17430607,1.62359422,16.2171337,49.20342242,90.52431235,29.19027778,944.87321779]])

#how to transform and inverse transform
#np.concatenate((a,b[:,None]),axis=1)

#y_prediction = np.concatenate((X_test,ynew[:,None]),axis=1)
"""
db_prediction = np.column_stack([X_test,prediction])
db_real = np.column_stack([X_test,y_test])

np.set_printoptions(suppress=True)
db_finalpred = scaler.inverse_transform(db_prediction)
db_finalreal = scaler.inverse_transform(db_real)

y_predicted = db_finalreal[:,17]
y_real = db_finalpred[:,17]

print(y_predicted)
print(y_real)
score = new_model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test real error:', score[1])
"""

