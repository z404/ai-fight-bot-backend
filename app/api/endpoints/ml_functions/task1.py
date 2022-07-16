# Program to perform knn on iris dataset
# KNN N neighbours
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

iris = load_iris()

X = PCA(n_components=2).fit_transform(iris.data)
y = iris.target

def task_1(parameters):
    if "n_neighbors" not in parameters:
        return (False, 0, "Invalid parameters", None)
    return __task_1_execute(parameters["n_neighbors"], parameters["identifier"] or "test")

def __task_1_execute(n_neighbors=75, identifier="test"):
    # Change n neighbours to a much lower number to increase accuracy
    if n_neighbors > 85:
        return (False, 0, "n_neighbors too high", None)
    elif n_neighbors <= 0:
        return (False, 0, "n_neighbors too low", None)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)

    plot_decision_regions(X=X, y=y, clf=knn, legend=2)
    # make images directory if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")
    plt.savefig(f"images/{identifier}_task_1.png")
    plt.clf()

    if acc_score > 0.8 and acc_score <= 0.95:
        # Progress
        return (False, acc_score, "You're getting close!", f"images/{identifier}_task_1.png")
    elif acc_score > 0.95:
        # Success
        return (True, acc_score, "Task 1 Complete!", f"images/{identifier}_task_1.png")
    else:
        # Failure
        return (False, acc_score, "Try again!", f"images/{identifier}_task_1.png")

def task_detail_1():
    return {
        "task_id": 1,
        "task_name": "To run KNN on iris dataset, by modifying the n_neighbors parameter",
        "parameters": {
            "n_neighbors": {
                "type": "int",
                "description": "Number of neighbours to use in KNN",
                "default": 75,
                "max": 85,
                "min": 0
            }
        }
    }