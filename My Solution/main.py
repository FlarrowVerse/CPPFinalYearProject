import numpy as np
from fcm_pso import *

# Define three cluster centers
centers = [[3, 2],
           [1, 7],
           [5, 6]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.8, 0.3],
          [0.3, 0.5],
          [1.1, 0.7]]

# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts = np.zeros(0)
ypts = np.zeros(0)
labels = np.zeros(0)

max_val = 100

for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts = np.hstack((xpts, np.random.standard_normal(max_val) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(max_val) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(max_val) * i))

alldata = np.vstack((xpts, ypts))
print(alldata.shape)
X = alldata.transpose()

fcm = FCM(n_clusters=3)
u = (fcm.fit(X, np.array(centers)))

print(u.shape)
print(u)