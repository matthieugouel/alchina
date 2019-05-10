"""Example of algorithm selection for multiclass classification."""

import pandas as pd

from alchina.classifiers import LinearClassifier
from alchina.metrics import precision_score, recall_score, f1_score
from alchina.selection import split_dataset
from alchina.utils import target_reshape

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from timeit import default_timer as timer


def train_and_report(model, X_train, y_train, X_test, y_test):
    """Train the model and print a report."""

    # Training
    start = timer()
    model.fit(X_train, y_train)
    end = timer()

    # Report
    print(f"Accuracy : {model.score(X_test, y_test):.3f}")
    print("---")
    precision = precision_score(model.predict(X_test), y_test, average="macro")
    print(f"Precision : {precision:.3f}")
    recall = recall_score(model.predict(X_test), y_test, average="macro")
    print(f"Recall : {recall:.3f}")
    print("---")
    f1 = f1_score(model.predict(X_test), y_test, average="macro")
    print(f"F1 score : {f1:.3f}")
    print("---")
    print(f"Training time : {end - start:.4f}s\n")


if __name__ == "__main__":
    # Load the Boston house prices dataset
    iris = datasets.load_iris()

    # Create the input features and target data frame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = pd.Series(iris.target)

    # Select the input features
    X = df[iris.feature_names].values

    # Select the target
    y = df["target"]
    y = y.values.reshape(y.shape[0], 1)

    X_train, y_train, X_test, y_test = split_dataset(X, y)

    # Alchina regression
    print("--- Alchina ---")
    lr_alchina = LinearClassifier(learning_rate=0.05, iterations=2000, standardize=True)
    train_and_report(lr_alchina, X_train, y_train, X_test, y_test)

    # Scikit-learn regression
    print("--- Scikit-learn ---")
    lr_skitlearn = LogisticRegression(solver="liblinear", multi_class="ovr")
    train_and_report(
        lr_skitlearn, X_train, target_reshape(y_train), X_test, target_reshape(y_test)
    )
