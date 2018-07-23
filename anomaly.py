print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)

X = np.array([

[8.80,0.27],
[105.98,2.00],
[20.00,0.63],
[175.00,0.00],
[4.40,0.13],
[22.65,1.93],
[5.00,0.14],
[4.40,0.19],
[17.32,0.00],
[28.10,1.53],
[32.14,1.72],
[20.82,1.18],
[22.08,1.24],
[24.16,1.34],
[15.50,0.92],
[44.00,1.09],
[62.88,2.61]


])

# fit the model
clf = LocalOutlierFactor(n_neighbors=40)
y_pred = clf.fit_predict(X)

X_plot = np.empty([0 , 2], dtype=float)

outlier_count = 0

# Outliers
for idx, predict in np.ndenumerate(y_pred):
  if predict == -1: 
    X_plot = np.concatenate((X_plot, [ X[idx] ] ))
    outlier_count = outlier_count + 1

# Not Outliers
for idx, predict in np.ndenumerate(y_pred):
  if predict == 1:
    X_plot = np.concatenate((X_plot, [ X[idx] ]))


print('Outlier Count: ', outlier_count)
print X_plot


# ploting
a = plt.scatter(X_plot[outlier_count:, 0], X_plot[outlier_count:, 1], c='white',
                edgecolor='k', s=20)
b = plt.scatter(X_plot[:outlier_count, 0], X_plot[:outlier_count, 1], c='red',
                edgecolor='k', s=20)
plt.axis('tight')

# graphic range
plt.xlim((-50, 1500))
plt.ylim((-5, 150))

plt.legend([a, b],
           ["normal observations",
            "abnormal observations"],
           loc="upper left")
plt.show()