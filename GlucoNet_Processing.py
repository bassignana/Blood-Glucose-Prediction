#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:48:45 2020

@author: tommasobassignana
"""
#TO DO: cap 10 per verifiche sul withe noise
#TO DO: cap 12 decomposition of time series
#TO DO: run all the models correctly at least 1 time
#TO DO: save and load NN models for evaluation and for retraining
#TO DO: plot training and test loss to evauate performance for every algorithm
#TO DO: train the neural networks with the 7 step procedure by Justin
#TO DO: datetime feature engineering
import pandas as pd
import pickle
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import uniform as sp_rand
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA

tscv = TimeSeriesSplit(n_splits=3)

def extract_training_data(data):
        """
        Extract the input variables (x), the time (t), and the objective (y) from the data samples.
        WARNING : need to be modified to include additional data, or override the function within the models
        :param data:
        :return:
        """
        
        y = data["y"]
        x = data.drop(["y", "datetime"], axis=1)

        return x, y

xtrain, ytrain = extract_training_data(train)
xtest, ytest = extract_training_data(test)

def extract_training_glucose(data):
        """
        Extract the input variables (x), the time (t), and the objective (y) from the data samples.
        WARNING : need to be modified to include additional data, or override the function within the models
        :param data:
        :return:
        """
        
        
        y = data["y"]
        x = data.filter(like='glucose',axis=1)
        return x, y

###########################################################
#time series
###########################################################
xtrain, ytrain = extract_training_glucose(train)
xtest, ytest = extract_training_glucose(test)

data_resampled.shape
X = data_resampled["glucose"].shift(6)
y = data_resampled["glucose"]
data_univ = pd.concat([X,y], axis=1, keys = ["X", "y"])
data_univ.dropna(inplace=True)

#autocorrelation check
lag_plot(data_univ["X"])
#c'è, e molta, notare gli assi del grafico

#autocorrelation plot
autocorrelation_plot(data_univ["X"])#già fatta altrove

#acf
plot_acf(data_univ["X"], lags=100)
#non è multistep.......
# # split dataset
# X = data_univ["X"]
# train, test = X[1:len(X)-7], X[len(X)-7:]

# # train autoregression#######
# model = AR(train)
# model_fit = model.fit()
# print('Lag: %s' % model_fit.k_ar)
# print('Coefficients: %s' % model_fit.params)
# # make predictions
# predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False) 

# rmse = sqrt(mean_squared_error(test, predictions))

# print('Test RMSE: %.3f' % rmse)
# # plot results
# plt.plot(test) 
# plt.plot(predictions, color='red') 
# plt.show()

####### ARIMA#####- problema è che non posso usarla con il prediction orizon
X = pd.Series(data_univ["X"])
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
test.reset_index(inplace=True, drop = True)
# walk-forward validation - LENTISSIMA perchè refitta il modello ogni volta per tener conto delle osservazioni passate
history = [x for x in train]
predictions = list()
real_values = list()
pred_h = 6
for t in range(len(test)):
  model = ARIMA(history, order=(5,1,0))
  model_fit = model.fit(disp=0)
  output = model_fit.forecast(steps = pred_h)
  yhat = output[0][pred_h-1]
  predictions.append(yhat)
  y = test[t+pred_h-1]
  real_values.append(y)
  obs = test[:t+pred_h]
  obs = obs.values
  history.extend(obs)
  print("stop")
  print('predicted=%f, expected=%f' % (yhat, y))
# evaluate forecasts
rmse = sqrt(mean_squared_error(real_values, predictions)) 
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes 
plt.plot(test)
plt.plot(predictions, color='red') 
plt.show()


#######




#stupid plot
# plt.figure()
# for i in range(len(data_resampled.columns)):
#   # create subplot
#   plt.subplot(len(data_resampled.columns), 1, i+1)
#   # get variable name
#   name = data_resampled.columns[i]
#   # create histogram
#   data_resampled[name].hist(bins=100)
#   # set title
#   plt.title(name, y=0, loc='right')
#   # turn off ticks to remove clutter
#   plt.yticks([])
#   plt.xticks([])
# plt.show()


       
# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
  scores = list()
  # calculate an RMSE score for each day
  for i in range(actual.shape[1]):
    # calculate mse
    mse = mean_squared_error(actual[:, i], predicted[:, i])
    # calculate rmse
    rmse = sqrt(mse)
    # store
    scores.append(rmse)
  # calculate overall RMSE
s=0
for row in range(actual.shape[0]):
    for col in range(actual.shape[1]):
      s += (actual[row, col] - predicted[row, col])**2
  score = sqrt(s / (actual.shape[0] * actual.shape[1]))
  return score, scores

#Running the function will first return the overall RMSE regardless of the period, then an array of RMSE scores for each period


# evaluate a single model
def evaluate_model(model_func, train, test):
  # history is a list of weekly data
  history = [x for x in train]
  # walk-forward validation over each week
  predictions = list()
  for i in range(len(test)):
    # predict the week
    yhat_sequence = model_func(history)
    # store the predictions
    predictions.append(yhat_sequence)
    # get real observation and add to history for predicting the next week
    history.append(test[i, :])
  predictions = array(predictions)
  # evaluate predictions days for each week
  score, scores = evaluate_forecasts(test[:, :, 0], predictions)
  return score, scores


# summarize scores
def summarize_scores(name, score, scores):
s_scores = ', '.join(['%.1f' % s for s in scores]) print('%s: [%.3f] %s' % (name, score, s_scores))
###########################################################
#time series fine
###########################################################
#
#train.to_csv("train_prova", index = False)
#test.to_csv("test_prova",index = False)

#multiple linear reg
from yellowbrick.regressor import ResidualsPlot
model = linear_model.LinearRegression()
visualizer = ResidualsPlot(model)

visualizer.fit(xtrain, ytrain)  # Fit the training data to the visualizer
visualizer.score(xtest, ytest)  # Evaluate the model on the test data
visualizer.show()


# Create linear regression object
model = linear_model.LinearRegression()

# Train the model using the training sets
model.fit(xtrain, ytrain)

# residual plots
residual = (ytest - model.predict(xtest))
residual.plot()
#buono perchè non ci sono trend evidenti non catturati dal modello 

#residual statistics
#se la media è vicino a zero è un forte suggerimento dell'assenza di bias nella predizione
residual.describe()

#residual histogram and density plots
#vorremmo una distribuzione normale con media zero
residual.plot(kind='kde')
residual.hist()
qqplot(residual, line='r')#la differenza sulle code è perchè la diistribuzione degli errori assomiglia ad una t stud
autocorrelation_plot(residual)


results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(model, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))

# save the model to disk
filename = 'multiple_linear_reg_subject_.sav'
with open('/Users/tommasobassignana/Desktop/GlucoNet/saved_models/'+filename, 'wb') as f:
    pickle.dump(model, f)
 
# some time later... 
# load the model from disk

with open('/Users/tommasobassignana/Desktop/GlucoNet/saved_models/'+filename, 'rb') as f:
    loaded_model = pickle.load(f)
result = loaded_model.score(xtest, ytest)
result

#create polynomial and interaction terms
#make subset selection

#ridge reg
#The complexity parameter alpha > 0 controls the amount of shrinkage: the larger the value of alpha, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.

model = linear_model.Ridge()
param_grid = {'alpha': sp_rand()}
# create and fit a ridge regression model, testing random alpha values
#model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, scoring="neg_mean_squared_error", cv = tscv, refit=True )
rsearch.fit(xtrain, ytrain)
model = rsearch.best_estimator_

results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(model, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))

# save the model to disk
filename = 'ridge_reg_subject_.sav'
with open('/Users/tommasobassignana/Desktop/GlucoNet/saved_models/'+filename, 'wb') as f:
    pickle.dump(model, f)


#Lasso
model = linear_model.Lasso()
param_grid = {'alpha': sp_rand()}

rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, scoring="neg_mean_squared_error", cv = tscv, refit=True )
rsearch.fit(xtrain, ytrain)
model = rsearch.best_estimator_

results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(model, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))

# save the model to disk
filename = 'lasso_reg_subject_.sav'
with open('/Users/tommasobassignana/Desktop/GlucoNet/saved_models/'+filename, 'wb') as f:
    pickle.dump(model, f)

#Elastic Net
from sklearn.linear_model import ElasticNetCV
modelcv = ElasticNetCV(cv=tscv, random_state=0)
modelcv.fit(xtrain,ytrain)


results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(modelcv, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))

# save the model to disk
filename = 'elasticNet_reg_subject_.sav'
with open('/Users/tommasobassignana/Desktop/GlucoNet/saved_models/'+filename, 'wb') as f:
    pickle.dump(modelcv, f)

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
#knn https://medium.com/@erikgreenj/k-neighbors-classifier-with-gridsearchcv-basics-3c445ddeb657
model = neighbors.KNeighborsRegressor()
param_grid  = {'n_neighbors':[1,2,5,10,20,35,50,100],'weights':('uniform', 'distance')}
#si possono provare anche altre distanze
gsearch = GridSearchCV(model, param_grid)
gsearch.fit(xtrain, ytrain)
gsearch.best_params_
k = list(gsearch.best_params_.values())[0]
w = list(gsearch.best_params_.values())[1]

model = neighbors.KNeighborsRegressor(k,weights = w)

results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(model, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))
   
# save the model to disk
filename = 'knn_reg_subject_.sav'
with open('/Users/tommasobassignana/Desktop/GlucoNet/saved_models/'+filename, 'wb') as f:
    pickle.dump(model, f)

from sklearn import tree
#decision tree
model = tree.DecisionTreeRegressor()
model.fit(xtrain, ytrain)

results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(model, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))

# save the model to disk
filename = 'decision_tree_reg_subject_.sav'
with open('/Users/tommasobassignana/Desktop/GlucoNet/saved_models/'+filename, 'wb') as f:
    pickle.dump(model, f)

#random forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
import time

start = time.time()

rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 10, cv = tscv, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(xtrain, ytrain)


end = time.time()
print(end - start)

best_random = rf_random.best_estimator_

results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(best_random, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))

# save the model to disk
filename = 'random_forest_subject_.sav'
with open('/Users/tommasobassignana/Desktop/GlucoNet/saved_models/'+filename, 'wb') as f:
    pickle.dump(best_random, f)

#extremely randomized trees
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()

n_trees = [10, 50, 100, 500, 1000, 5000]
max_feature = [int(x) for x in np.linspace(start = 1, stop = 3, num = 3)]
min_samples_split = [int(x) for x in np.linspace(start = 1, stop = 20, num = 20)]

random_grid = {'n_estimators': n_trees,
               'max_features': max_feature,
               'min_samples_split': min_samples_split}

extra = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 1, cv = tscv, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
extra.fit(xtrain, ytrain)

best_extra = extra.best_estimator_

results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(best_extra, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))

# save the model to disk
filename = 'random_forest_subject_.sav'
with open('/Users/tommasobassignana/Desktop/GlucoNet/saved_models/'+filename, 'wb') as f:
    pickle.dump(best_extra, f)

#Adaboost
from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()

random_grid = {
 'n_estimators': [50, 100],
 'learning_rate' : [0.01,0.05,0.1,0.3,1],
 'loss' : ['linear', 'square', 'exponential']
 }

ada = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 1, cv = tscv, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
ada.fit(xtrain, ytrain)

best_ada = extra.best_estimator_

results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(best_ada, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))

# save the model to disk
filename = 'random_forest_subject_.sav'
with open('/Users/tommasobassignana/Desktop/GlucoNet/saved_models/'+filename, 'wb') as f:
    pickle.dump(best_ada, f)

# #multilayer perceptron regression
# from sklearn.neural_network import MLPRegressor
# regr_mlp = MLPRegressor()
# # Train the model using the training sets
# regr_mlp.fit(x, y)
# # Make predictions using the testing set
# ypreds = regr_mlp.predict(xtest)
# # mean squared error
# mean_squared_error(ytest, ypreds)

######################################################################
#reshape for lstm
######################################################################

xtrain.values.shape
xtrain_lstm = xtrain.values.reshape((xtrain.shape[0],xtrain.shape[1],1))
xtrain_lstm.shape#ok



#need splits of 200-400 obs ?????????????????
# xtrain_lstm = xtrain[:-23]
# xtrain_lstm.shape

# xtrain_lstm_samples = list()
# samples = list()
# length = 200
# # step over the 5,000 in jumps of 200
# for i in range(0,xtrain_lstm.shape[0],length):
#   # grab from i to i + 200
#   sample = xtrain_lstm[i:i+length]
#   samples.append(sample)
# # convert list of arrays into 2d array
# xtrain_lstm_samples = np.array(samples)
# # reshape into [samples, timesteps, features]
# xtrain_lstm_samples = xtrain_lstm_samples.reshape((len(samples), length, 1))
# print(xtrain_lstm_samples.shape)




# # define the dataset
# data = list()
# n = 5000
# for i in range(n):
#   data.append([i+1, (i+1)*10])
# data = array(data)
# # drop time
# data = data[:, 1]
# # split into samples (e.g. 5000/200 = 25)
# samples = list()
# length = 200
# # step over the 5,000 in jumps of 200
# for i in range(0,n,length):
#   # grab from i to i + 200
#   sample = data[i:i+length]
#   samples.append(sample)
# # convert list of arrays into 2d array
# data = np.array(samples)
# # reshape into [samples, timesteps, features]
# data = data.reshape((len(samples), length, 1))
# print(data.shape)
######################################################################
#reshape for lstm fine
######################################################################

######################################################################
#basic univariate lstm models
######################################################################
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

n_steps = xtrain_lstm.shape[1]
n_features = 1


# define model vanilla 
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(n_steps,
n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(xtrain_lstm, ytrain, epochs=20, verbose=1)

yhat = model.predict(x_input, verbose=0)




# define model stacked 
model = Sequential()
model.add(LSTM(20, activation='relu', return_sequences=True, input_shape=(n_steps,
n_features)))
model.add(LSTM(20, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(xtrain_lstm, ytrain, epochs=20, verbose=1)

yhat = model.predict(x_input, verbose=0)




# define model bidirectional
model = Sequential()
model.add(Bidirectional(LSTM(20, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(xtrain_lstm, ytrain, epochs=20, verbose=1)

yhat = model.predict(x_input, verbose=0)

#CNN LSTM - non va
# reshape from [samples, timesteps(n_steps)] into [samples, subsequences, timesteps_steps, features]
xtrain_CNN_LSTM = xtrain_lstm[:-23]
xtrain_CNN_LSTM.shape
div=2
xtrain_CNN_LSTM = xtrain_CNN_LSTM.reshape(xtrain_CNN_LSTM.shape[0], div, int(xtrain_CNN_LSTM.shape[1]/div),1)

xtrain_CNN_LSTM.shape
n_features = 1
# define model
model = Sequential()

#changed filters from 64 to 12 to make it work
model.add(TimeDistributed(Conv1D(filters=12, kernel_size=1, activation='relu'),
input_shape=(None, int(xtrain_CNN_LSTM.shape[1]/div), n_features)))

#changed from pool size 2 to pool size 1
model.add(TimeDistributed(MaxPooling1D(pool_size=1)))

model.add(TimeDistributed(Flatten())) 

model.add(LSTM(20, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse') # fit model

#ytrain rimane uguale
model.fit(xtrain_CNN_LSTM, ytrain[:-23], epochs=20, verbose=1)





######################################################################
#basic univariate lstm models fine
######################################################################



















import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
print(tf.__version__)

#model building via custom function
def build_model():
  model = keras.Sequential([
    layers.Dense(72, activation='relu'),
    layers.Dense(72, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

history = model.fit(
  xtrain, ytrain,
  epochs=EPOCHS, validation_split = 0.2, verbose=1, )
#loss, mae, mse
model.evaluate(xtest, ytest, verbose=2)

######################################################################
import matplotlib.pyplot as plt
#single step model - LINEAR
def build_model():
  model = linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

EPOCHS = 20
model = build_model()

history = model.fit(
  xtrain, ytrain,
  epochs=EPOCHS, validation_split = 0.2, verbose=1, )
#loss, mae, mse
model.evaluate(xtest, ytest, verbose=2)

ypreds = model.predict(xtest)
#PLOT
print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
######################################################################
#multi_step_dense

def build_model():
  model  = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=72, activation='relu'),
    tf.keras.layers.Dense(units=72, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

EPOCHS = 20
model = build_model()

history = model.fit(
  xtrain, ytrain,
  epochs=EPOCHS, validation_split = 0.2, verbose=1, )
#loss, mae, mse
model.evaluate(xtest, ytest, verbose=2)
######################################################################
#conv net

def build_model():
  model  =  tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=72,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=72, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

EPOCHS = 20
model = build_model()

history = model.fit(
  xtrain, ytrain,
  epochs=EPOCHS, validation_split = 0.2, verbose=1, )
#loss, mae, mse
model.evaluate(xtest, ytest, verbose=2)






from numpy import array
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
in_seq1 = array(train_df["glucose"])
in_seq2 = array(train_df["CHO"])
in_seq3 = array(train_df["insulin"])
out_seq = array(train_df["glucose"])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, in_seq3, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
# summarize the data
# for i in range(len(X)):
# 	print(X[i], y[i])
# The first dimension is the number of samples, in this case 8450. The second dimension is the number of time steps per sample, in this case 3, the value specified to the function. Finally, the last dimension specifies the number of parallel time series or the number of variables, in this case 3 for the two parallel series.
# This is the exact three-dimensional structure expected by a 1D CNN as input. The data is ready to use without further reshaping.
n_features = X.shape[2]

from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
def build_model():
  model = Sequential()
  model.add(Conv1D(filters=64, kernel_size=3, activation='relu',                    input_shape=(n_steps, n_features)))
  model.add(MaxPooling1D(pool_size=3,padding='same'))
  model.add(Flatten())
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1))

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

EPOCHS = 20
model = build_model()


in_seq1 = array(test_df["glucose"])
in_seq2 = array(test_df["CHO"])
in_seq3 = array(test_df["insulin"])
out_seq = array(test_df["glucose"])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, in_seq3, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
Xtest, Ytest = split_sequences(dataset, n_steps)

history = model.fit(
  X, y,
  epochs=EPOCHS, validation_split = 0.2, verbose=1, )
#loss, mae, mse
model.evaluate(xtest, ytest, verbose=2)



######################################################################

#LSTM
def build_model():
  model = linear = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

EPOCHS = 20
model = build_model()

history = model.fit(
  xtrain, ytrain,
  epochs=EPOCHS, validation_split = 0.2, verbose=1, )
#loss, mae, mse
model.evaluate(xtest, ytest, verbose=2)


