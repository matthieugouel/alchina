"""Example of unsupervized learning using K-means algorithm."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alchina.clusters import KMeans

from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()

# Create the input features
df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
df["target"] = pd.Series(iris.target)

# Select the input features
X = df[["sepal length (cm)", "petal length (cm)"]]
X = X.values

# Train the model
kmeans = KMeans(n_centroids=2)
kmeans.fit(X)

# Display the number of model iterations
print(f"K-Means iterations : {kmeans.iterations}")

# Plot the results
A = X[np.array(kmeans.indexes) == 0]
B = X[np.array(kmeans.indexes) == 1]

plt.scatter(A[:, 0], A[:, 1], color="blue")
plt.scatter(B[:, 0], B[:, 1], color="red")

plt.show()
