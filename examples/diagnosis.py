"""Example of algorithm diagnosis."""

import pandas as pd

from alchina.diagnosis import split_dataset
from alchina.regressors import LinearRegressor

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from timeit import default_timer as timer


# Load the Boston house prices dataset
houses = datasets.load_boston()

# Create the input features and target data frame
df = pd.DataFrame(houses.data, columns=houses.feature_names)
df["target"] = pd.Series(houses.target)

# Select the input features
X = df[houses.feature_names].values

# Select the target
y = df["target"]
y = y.values.reshape(y.shape[0], 1)

X_train, y_train, X_test, y_test = split_dataset(X, y)

# Alchina regression
print("--- Alchina ---")
start = timer()
lr_alchina = LinearRegressor(learning_rate=0.03, iterations=1000, standardize=True)
lr_alchina.fit(X_train, y_train)
end = timer()

print(f"Regression score : {lr_alchina.score(X_test, y_test):.3f}")
print(f"Computation time : {end - start:.4f}s")

# Scikit-learn regression
print("--- Scikit-learn ---")
start = timer()
lr_skitlearn = LinearRegression(normalize=True)
lr_skitlearn.fit(X_train, y_train)
end = timer()

print(f"Regression score : {lr_skitlearn.score(X_test, y_test):.3f}")
print(f"Computation time : {end - start:.4f}s")
