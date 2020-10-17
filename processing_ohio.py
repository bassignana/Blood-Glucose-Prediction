#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import uniform as sp_rand
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
# def extract_training_data(data, cv_index):
#         """
#         Extract the input variables (x), the time (t), and the objective (y) from the data samples.
#         WARNING : need to be modified to include additional data, or override the function within the models
#         :param data:
#         :return:
#         """
#         data = data[cv_index]
        
#         t = data["datetime"]
#         y = data["y"]
#         x = data.drop(["y", "datetime", "time"], axis=1)

#         return x, y, t

# x, y, t = extract_training_data(train, 0)

# def extract_training_glucose(data, cv_index):
#         """
#         Extract the input variables (x), the time (t), and the objective (y) from the data samples.
#         WARNING : need to be modified to include additional data, or override the function within the models
#         :param data:
#         :return:
#         """
#         data = data[cv_index]
        
#         t = data["datetime"]
#         y = data["y"]
#         x = data.filter(like='glucose',axis=1)
#         return x, y, t


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






#multiple linear reg
# Create linear regression object
model = linear_model.LinearRegression()

# Train the model using the training sets
model.fit(xtrain, ytrain)

results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(model, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))

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


#Elastic Net
from sklearn.linear_model import ElasticNetCV
modelcv = ElasticNetCV(cv=tscv, random_state=0)
modelcv.fit(xtrain,ytrain)


results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(model, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))



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

from sklearn import tree
#decision tree
model = tree.DecisionTreeRegressor()
model.fit(xtrain, ytrain)

results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(model, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))

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
rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 1, cv = tscv, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(xtrain, ytrain)

best_random = rf_random.best_estimator_

results = []
metrics=["max_error","neg_root_mean_squared_error","neg_mean_squared_error","neg_mean_absolute_error"]
    
for metric in metrics:
   results.append(cross_val_score(best_random, X = xtest, y = ytest, cv=tscv, verbose=0, scoring=metric))


# #extremely randomized trees
# from sklearn.ensemble import ExtraTreesRegressor
# regr_ext = ExtraTreesRegressor()
# # Train the model using the training sets
# regr_ext.fit(x, y)
# # Make predictions using the testing set
# ypreds = regr_ext.predict(xtest)
# # mean squared error
# mean_squared_error(ytest, ypreds)

#Adaboost
from sklearn.ensemble import AdaBoostRegressor
regr_ada = AdaBoostRegressor()
# Train the model using the training sets
regr_ada.fit(x, y)
# Make predictions using the testing set
ypreds = regr_ada.predict(xtest)
# mean squared error
mean_squared_error(ytest, ypreds)

#multilayer perceptron regression
from sklearn.neural_network import MLPRegressor
regr_mlp = MLPRegressor()
# Train the model using the training sets
regr_mlp.fit(x, y)
# Make predictions using the testing set
ypreds = regr_mlp.predict(xtest)
# mean squared error
mean_squared_error(ytest, ypreds)

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
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

#check
example_batch = x[:10]
example_result = model.predict(example_batch)
example_result

#training the model
EPOCHS = 10

history = model.fit(
  x, y,
  epochs=EPOCHS, validation_split = 0.2, verbose=0, )
#loss, mae, mse
model.evaluate(xtest, ytest, verbose=2)

ypreds = model.predict(xtest)
# mean squared error
mean_squared_error(ytest, ypreds)
