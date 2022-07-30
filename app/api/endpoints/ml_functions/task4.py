from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_4(parameters):
    if "min_samples_leaf" not in parameters:
        return (False, 0, " provided", None)
    return __task_4_execute__(min_samples_leaf=parameters["min_samples_leaf"], identifier = parameters["identifier"] or "test")

def __task_4_execute__(min_samples_leaf=7, identifier="test"):

    if min_samples_leaf > 20:
        return (False, 0, "min samples leaf too high", None)
    elif min_samples_leaf <= 0:
        return (False, 0, "min samples leaf too low", None)

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

    tree_clf = DecisionTreeClassifier(random_state=42, criterion= 'entropy', max_depth=7, min_samples_leaf= min_samples_leaf, min_samples_split= 2, splitter= 'best', max_features=9)
    tree_clf.fit(X_train, y_train)

    y_pred = tree_clf.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)

    plot_tree(tree_clf, feature_names=X_train.columns, class_names=['Not Attrition', 'Attrition'], filled=True)
    plt.savefig(f"images/{identifier}_task_4.png", bbox_inches='tight')
    plt.clf()

    if acc_score >= 0.865 and acc_score <= 1:
        # Success
        return (True, acc_score, "Task 4 Complete!", f"images/{identifier}_task_4.png")
    elif acc_score >= 0.855 and acc_score < 0.865:
        # Progress
        return (False, acc_score, "You're getting close!", f"images/{identifier}_task_4.png")
    else:
        # Failure
        return (False, acc_score, "Try again!", f"images/{identifier}_task_4.png")

# print(task_4(min_samples_leaf=8, identifier="test"))

def task_detail_4():
    return {
        "task_id": 4,
        "task_name": "To run the Decision Tree Classifier on the HR Employee Attrition dataset, by modifying the min_samples_leaf parameter.",
        "parameters": {
            "min_samples_leaf": {
                "type": "int",
                "description": "The minimum number of samples required to be at a leaf node.",
                "default": 7,
                "min": 1,
                "max": 20
            }
        }
    }
