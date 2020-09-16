#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def extract_training_data(data, cv_index):
        """
        Extract the input variables (x), the time (t), and the objective (y) from the data samples.
        WARNING : need to be modified to include additional data, or override the function within the models
        :param data:
        :return:
        """
        data = data[cv_index]
        
        t = data["datetime"]
        y = data["y"]
        x = data.drop(["y", "datetime", "time"], axis=1)

        return x, y, t

x, y, t = extract_training_data(train, 0)

def extract_training_glucose(data, cv_index):
        """
        Extract the input variables (x), the time (t), and the objective (y) from the data samples.
        WARNING : need to be modified to include additional data, or override the function within the models
        :param data:
        :return:
        """
        data = data[cv_index]
        
        t = data["datetime"]
        y = data["y"]
        x = data.filter(like='glucose',axis=1)
        return x, y, t

x, y, t = extract_training_glucose(train, 0)

xtest, ytest, ttest = extract_training_glucose(test, 0)
#multiple linear reg
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(x, y)
# Make predictions using the testing set
ypreds = regr.predict(xtest)
# mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(ytest, ypreds))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(ytest, ypreds))

#ridge reg
#The complexity parameter alpha > 0 controls the amount of shrinkage: the larger the value of alpha, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.

# Create linear regression object
regr_ridge = linear_model.Ridge(alpha=.1)
# Train the model using the training sets
regr_ridge.fit(x, y)
# Make predictions using the testing set
ypreds = regr_ridge.predict(xtest)
# mean squared error
mean_squared_error(ytest, ypreds)

#Lasso
# Create linear regression object
regr_lasso = linear_model.Lasso(alpha=.1)
# Train the model using the training sets
regr_lasso.fit(x, y)
# Make predictions using the testing set
ypreds = regr_lasso.predict(xtest)
# mean squared error
mean_squared_error(ytest, ypreds)

#elastic net
regr_el = linear_model.ElasticNet(alpha=0.5, l1_ratio=0.7)
# Train the model using the training sets
regr_el.fit(x, y)
# Make predictions using the testing set
ypreds = regr_el.predict(xtest)
# mean squared error
mean_squared_error(ytest, ypreds)

from sklearn import neighbors
#knn
regr_knn = neighbors.KNeighborsRegressor(5, weights="uniform")
# Train the model using the training sets
regr_knn.fit(x, y)
# Make predictions using the testing set
ypreds = regr_knn.predict(xtest)
# mean squared error
mean_squared_error(ytest, ypreds)

from sklearn import tree
#decision tree
regr_tree = tree.DecisionTreeRegressor()
# Train the model using the training sets
regr_tree.fit(x, y)
# Make predictions using the testing set
ypreds = regr_tree.predict(xtest)
# mean squared error
mean_squared_error(ytest, ypreds)

#random forest
from sklearn.ensemble import RandomForestRegressor
regr_rf = RandomForestRegressor()
# Train the model using the training sets
regr_rf.fit(x, y)
# Make predictions using the testing set
ypreds = regr_rf.predict(xtest)
# mean squared error
mean_squared_error(ytest, ypreds)

#extremely randomized trees
from sklearn.ensemble import ExtraTreesRegressor
regr_ext = ExtraTreesRegressor()
# Train the model using the training sets
regr_ext.fit(x, y)
# Make predictions using the testing set
ypreds = regr_ext.predict(xtest)
# mean squared error
mean_squared_error(ytest, ypreds)

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
