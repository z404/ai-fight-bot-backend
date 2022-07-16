# Program to perform knn on iris dataset

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

def task_2(parameters):
    if "distance_metric" not in parameters:
        return (False, 0, "Invalid parameters", None)
    return __task_2_execute(parameters["distance_metric"], parameters["identifier"] or "test")

def __task_2_execute(distance_metric='manhattan', identifier="test"):
    if distance_metric not in ['euclidean', 'manhattan', 'cosine', 'minkowski']:
        return (False, 0, "Invalid distance metric", None)
    # Change n neighbours to a much lower number to increase accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=7, metric=distance_metric, p=1)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)

    plot_decision_regions(X=X, y=y, clf=knn, legend=2)
    # make images directory if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")
    plt.savefig(f"images/{identifier}_task_2.png")
    plt.clf()

    if acc_score > 0.8 and acc_score <= 0.95:
        # Progress
        return (False, acc_score, "You're getting close!", f"images/{identifier}_task_2.png")
    elif acc_score > 0.95:
        # Success
        return (True, acc_score, "Task 2 Complete!", f"images/{identifier}_task_2.png")
    else:
        # Failure
        return (False, acc_score, "Try again!", f"images/{identifier}_task_2.png")

def task_detail_2():
    return {
        "task_id": 2,
        "task_name": "To run KNN on iris dataset, by modifying the distance metric",
        "parameters": [
            {
                "name": "distance_metric",
                "type": "single_choice",
                "description": "The distance metric to use",
                "options": ["euclidean", "manhattan", "cosine", "minkowski"],
                "default": "manhattan"
            }
        ]
    }
