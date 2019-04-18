"""Example of logistic regression."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alchina.preprocessors import Standardization
from alchina.regressors import LogisticRegressor

from sklearn import datasets

# Load the dataset
cancer = datasets.load_breast_cancer()

# Select the input feature and target
df = pd.DataFrame(data=cancer["data"], columns=cancer["feature_names"])
df["target"] = pd.Series(cancer.target)

X = df[["mean radius", "mean texture"]]
X = X.values

y = df["target"]
y = y.values.reshape(X.shape[0], 1)

# Normalize the features
standardize = Standardization()
X = standardize(X)

# Perform the logistic regression
lr = LogisticRegressor(iterations=10000)
lr.fit(X, y)

# Plot the results
figure, (a0, a1) = plt.subplots(
    1,
    2,
    num="Logistic Regression",
    figsize=(16, 9),
    gridspec_kw={"width_ratios": [3, 1]},
)
figure.suptitle("Logistic Regression on breast cancer Wisconsin dataset", fontsize=16)

M = X[df["target"] == 0]
B = X[df["target"] == 1]

a0.scatter(M[:, 0], M[:, 1], color="red", label="Malignant tumors")
a0.scatter(B[:, 0], B[:, 1], color="green", label="Benign tumors")
a0.set(
    xlabel="Radius (mean of distances from center to points on the perimeter)",
    ylabel="Texture (standard deviation of gray-scale values)",
)

x_boundary = np.array([np.min(X[:, 0]) + 1, np.max(X[:, 0]) - 3])
y_boundary = (-1 / lr.parameters[2]) * (
    lr.parameters[1] * x_boundary + lr.parameters[0]
)

a0.plot(x_boundary, y_boundary, label="Decision Boundary")
a0.legend(loc="upper right")

a1.plot(range(len(lr.history)), lr.history)
a1.set(xlabel="Number of iterations", ylabel="Training error objective")

plt.show()
