#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#da Hands -On Time Series Analysis with Python by B V Vishwas, ASHISH PATEL (z-lib.org).pdf

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing 
import matplotlib.pyplot as plt

df = glucose_df_resampled
df.describe()
#voglio fare il forecast di tutti i valori nei 15 minuti successivi

#validation set
validate = df['glucose'].tail(9)
df.drop(df['glucose'].tail(200).index, inplace=True)

#correct format is a dataframe with one column and a time index
#Let’s rescale the data because neural networks are known to converge sooner with better accuracy when features are on the same scale.
df = df.values
scaler_x = preprocessing.MinMaxScaler()
x_rescaled = scaler_x.fit_transform(df.reshape(-1, 1))

#Define a function to prepare univariate data that is suitable for a time series.

def custom_ts_univariate_data_prep(dataset, start, end, window, horizon):
    X = []
    y = []
    start = start + window 
    if end is None:
        end = len(dataset) - horizon 
    for i in range(start, end):
        indicesx = range(i-window, i) 
        X.append(np.reshape(dataset[indicesx], (window, 1))) 
        indicesy = range(i,i+horizon) 
        y.append(dataset[indicesy])
    return np.array(X), np.array(y)


#qui sto dicedo che voglio usre una finestra di 24 valori per prevederne uno.
univar_hist_window = 24
horizon = 1
TRAIN_SPLIT = 1500

x_train_uni, y_train_uni = custom_ts_univariate_data_prep(x_rescaled, 0, TRAIN_SPLIT, univar_hist_window, horizon)

x_val_uni, y_val_uni = custom_ts_univariate_data_prep (x_rescaled, TRAIN_SPLIT, None,univar_hist_window,horizon)


#Prepare the training and validation time-series data using the tf.data
BATCH_SIZE = 256 
BUFFER_SIZE = 150

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

#The best weights are stored at model_path. 
model_path = r"/Users/tommasobassignana/Downloads/hands-on-time-series-analylsis-python-master/Code and Data/Chapter 5/LSTM_Univarient_1.h5"

#Define the LSTM model
lstm_model = tf.keras.models.Sequential([ tf.keras.layers.LSTM(100, input_shape=x_train_uni.shape[-2:],return_sequences=True),
tf.keras.layers.Dropout(0.2), tf.keras.layers.LSTM(units=50,return_sequences=False), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(units=1),
])

lstm_model.compile(optimizer='adam', loss='mse')
#Configure the model and start training with early stopping and checkpointing.
#Early stopping stops training when monitored loss starts to increase above patience.
#Checkpointing saves model weights as it reached minimum loss. 

EVALUATION_INTERVAL = 100
EPOCHS = 150
history = lstm_model.fit(train_univariate, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_univariate, validation_steps=50, verbose =1, callbacks =[tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min'), tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_ loss', save_best_only=False, mode='min', verbose=0)])


#Load the best weights into the model.
Trained_model = tf.keras.models.load_model(model_path)
#Check the model summary.
Trained_model.summary()
#Plot the loss and val_loss against the epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper left') 
plt.rcParams["figure.figsize"] = [16,9]
plt.show()

#dobiamo però prevedere 3 step nel futuro
#but our model predicts only one time step(horizon = 1), so let’s take the last 24 periods of data from the training and increment one prediction at a time and forecast the next values.

#serve il dataframe? si, df deve essere un dataframe. errore nel libro?
uni = df["glucose"]
validatehori = uni.tail(24)
validatehist = validatehori.values
result = []
# Define Forecast length here
window_len = 3
val_rescaled = scaler_x.fit_transform(validatehist.reshape(-1, 1))

for i in range(1, window_len+1):
    
    val_rescaled = val_rescaled.reshape((1, val_rescaled.shape[0], 1))
    Predicted_results = Trained_model.predict(val_rescaled)
    print(f'predicted : {Predicted_results}')
    result.append(Predicted_results[0])
    val_rescaled = np.append(val_rescaled[:,1:],[[Predicted_results]])
    print(val_rescaled)

#Rescale the predicted values back to the original scale.
#there is a problem with the result shape...
type(result)
result_inv_trans = scaler_x.inverse_transform(result)

from sklearn import metrics
def timeseries_evaluation_metrics_func(y_true, y_pred):
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')

timeseries_evaluation_metrics_func(validate,result_inv_trans)


timeseries_evaluation_metrics_func(validate,result)










