from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv("datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()

df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis="columns", inplace=True)

categorical_col = []
for column in df.columns:
    if df[column].dtype == object and len(df[column].unique()) <= 50:
        categorical_col.append(column)
        
df['Attrition'] = df.Attrition.astype("category").cat.codes

categorical_col.remove('Attrition')


label = LabelEncoder()
for column in categorical_col:
    df[column] = label.fit_transform(df[column])

X = df.drop('Attrition', axis=1)
y = df.Attrition

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def task_3(parameters):
    if "max_depth" not in parameters:
        return (False, 0, "Invalid parameters", None)
    return __task_3_execute(parameters["max_depth"], parameters["identifier"] or "test")

def __task_3_execute(max_depth=8, identifier="test"):

    if max_depth > 20:
        return (False, 0, "max_depth too high", None)
    elif max_depth <= 0:
        return (False, 0, "max_depth too low", None)

    tree_clf = DecisionTreeClassifier(random_state=42, criterion= 'entropy', max_depth=max_depth, min_samples_leaf= 10, min_samples_split= 2, splitter= 'best', max_features=3)
    tree_clf.fit(X_train, y_train)

    y_pred = tree_clf.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)

    # plt.figure(figsize=(5, 5))
    plot_tree(tree_clf, feature_names=X_train.columns, class_names=['Not Attrition', 'Attrition'], filled=True, )
    # make images directory if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")
    plt.savefig(f"images/{identifier}_task_3.png", bbox_inches='tight')
    plt.clf()

    if acc_score > 0.855 and acc_score <= 1:
        # Success
        return (True, acc_score, "Task 3 Complete!", f"images/{identifier}_task_3.png")
    elif acc_score > 0.84 and acc_score <= 0.855:
        # Progress
        return (False, acc_score, "You're getting close!", f"images/{identifier}_task_3.png")
    else:
        # Failure
        return (False, acc_score, "Try again!", f"images/{identifier}_task_3.png")

# print(task_3(7))
def task_detail_3():
    return {
        "task_id": 3,
        "task_name": "To run Decision tree algorithm on Employee Attrition Dataset, by modifying the max_depth parameter",
        "parameters": {
            "max_depth": {
                "type": "int",
                "description": "The maximum depth of the tree. The maximum depth limits the number of nodes in the tree. Must be greater than 0.",
                "default": 8,
                "max": 20,
                "min": 1
            }
        }
    } 