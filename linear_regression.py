# Code source: Jaques Grobler
# License: BSD 3 clause
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into training/testing sets
payments_X_train = [ 

	# Field 1: Total Value System 1
	# Field 2: Fee Value System 1
	# Field 3: Total Value System 2
	# Field 4: Fee Value System 2

	# Valids
	[100., 5., 100., 5.], 
	[50., 2.5, 50., 2.5], 
	[10., 0.5, 10., 0.5], 
	[5., 3., 5., 3.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],
	[100., 5., 100., 5.],

	# Invalids
	[50., 2.5, 0., 0],				# Total and Fee are diff between Systems
	[50., 2.5, 50., 25], 			# Fee is diff between Systems
	[50., 2.5, 50., 2.6],			# Fee is diff between Systems
	[500., 25., 500., 26.], 	# Fee is diff between Systems
	[100., 0, 100., 0],				# Fee is zero
	[100., -1., 100., -1.],		# Fee is negative
	[3., 5., 3., 5.],					# Fee is greather than Total
	[13., 40., 13., 40.],			# Fee is greather than Total
	[13., 40., 13., 40.],			# Fee is greather than Total
	[13., 40., 13., 40.],			# Fee is greather than Total
	[13., 40., 13., 40.],			# Fee is greather than Total
]

# Split the targets into training/testing sets
payments_y_train = [ 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ]


# Create linear regression object
regr = linear_model.LinearRegression()

# TRAIN the model using the training sets
regr.fit(payments_X_train, payments_y_train)


# TEST
payments_X_test = [ 
										# Valids
										[5., 3., 5., 3.], 
										[100., 5., 100., 5.],
										[100., 5., 100., 5.],

										# Invalids
										[11., 0.5, 11., 0.8],
										[11., -1., 11., -1.],
										[7., 10., 7., 10.]
									]
payments_y_test = [ 1., 1., 1., 0., 0., 0. ]


# Make PREDICTIONS using the testing set
payments_y_pred = regr.predict(payments_X_test)
print('Expected:', payments_y_test)
print('Predict:', payments_y_pred)


# The coefficients
print('Coefficients:', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(payments_y_test, payments_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(payments_y_test, payments_y_pred))




plt.scatter(payments_X_test, payments_y_test,  color='black')
plt.plot(payments_X_test, payments_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()