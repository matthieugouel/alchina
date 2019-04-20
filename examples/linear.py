"""Example of linear regression."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alchina.regressors import LinearRegressor

from sklearn import datasets


# Load the boston toy dataset
houses = datasets.load_boston()

# Create the features data frame
df = pd.DataFrame(houses.data, columns=houses.feature_names)
df["target"] = pd.Series(houses.target)

# Select the input feature
X = df["RM"]
X = X.values.reshape(X.shape[0], 1)

# Select the target
y = df["target"]
y = y.values.reshape(y.shape[0], 1)

# Train the model
lr = LinearRegressor(
    learning_rate=0.05, iterations=1000, history=True, standardize=True
)
lr.fit(X, y)

# Plot the results
figure, (a0, a1) = plt.subplots(
    1, 2, num="Linear Regression", figsize=(16, 9), gridspec_kw={"width_ratios": [3, 1]}
)
figure.suptitle("Linear Regression on Boston house prices dataset", fontsize=16)

a0.plot(X, y, "b.")
a0.plot(
    range(3, 10),
    [lr.predict(np.array([[x]])).flat[0] for x in range(3, 10)],
    "r-",
    label="Linear regression",
)
a0.set(
    xlabel="Average number of rooms per dwelling",
    ylabel="Median value of owner-occupied homes in $1000's",
)
a0.legend(loc="upper left")

a1.plot(range(len(lr.history)), lr.history)
a1.set(xlabel="Number of iterations", ylabel="Training error objective")

plt.show()
