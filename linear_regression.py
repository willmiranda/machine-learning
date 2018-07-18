# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Split the data into training/testing sets
# payments_X_train = payments_X[:-20]
payments_X_train = [ 

	# Valids
	[100., 5., 100., 5.], 
	[50., 2.5, 50., 2.5], 
	[10., 0.5, 10., 0.5], 

	# Invalids
	[50., 2.5, 50., 25], 
	[50., 2.5, 50., 2.6],
	[500., 25., 500., 26.], 
	[50., 2.5, 0., 0] ]

payments_X_test = [ [11., 0.5, 11., 0.8] ]

# Split the targets into training/testing sets
payments_y_train = [ 1., 1., 1., 0., 0., 0., 0. ]

payments_y_test = [ 0. ]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(payments_X_train, payments_y_train)

# Make predictions using the testing set
payments_y_pred = regr.predict(payments_X_test)
print('Expected:', payments_y_test, 'Predict: \n', payments_y_pred)


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(payments_y_test, payments_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(payments_y_test, payments_y_pred))