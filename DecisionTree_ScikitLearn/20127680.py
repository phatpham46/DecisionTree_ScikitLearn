import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, euclidean_distances, precision_recall_curve, auc, roc_curve
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
col_names = ["a1", "a2", "a3", "a4", "a5", "a6", "b1", "b2", "b3", "b4", "b5", "b6", "c1", "c2", "c3", "c4", "c5", "c6", "d1", "d2", "d3",
             "d4", "d5", "d6", "e1", "e2", "e3", "e4", "e5", "e6", "f1", "f2", "f3", "f4", "f5", "f6", "g1", "g2", "g3", "g4", "g5", "g6", "class"]

df = pd.read_csv('../lab2/input/connect-4.data', names=col_names)
labelencoder = LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
X = df.drop(['class'], axis=1)
y = df['class']


def splitData(p_test):
    feature_train, feature_test, label_train, label_test = train_test_split(
        X, y, test_size=p_test, random_state=0, shuffle=True, stratify=y)
    return feature_train, feature_test, label_train, label_test


def buildDecisionTree(feature_train, label_train, depth, p_test):
    clf = tree.DecisionTreeClassifier(
        criterion="gini", random_state=42, min_samples_leaf=43)
    labelencoder = LabelEncoder()
    for column in df.columns:
        df[column] = labelencoder.fit_transform(df[column])

    if depth == 0:
        clf = DecisionTreeClassifier()
    else:
        clf = DecisionTreeClassifier(max_depth=depth)

    clf = clf.fit(feature_train, label_train)
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(clf,
                       feature_names=X.columns,
                       filled=True, rounded=True, fontsize=4)
    plt.savefig("tree" + str((1-p_test)*100) + "_" + str(p_test*100) + ".png")
    return clf


def classificateReport(clf, feature_test, label_test):
    label_pred = clf.predict(feature_test)
    print("Decision Tree Classifier report \n",
          classification_report(label_test, label_pred))
    return


def confusionMatrix(feature_test, label_test, p_test):
    cfm = confusion_matrix(label_test, clf.predict(feature_test))
    sns.heatmap(cfm, annot=True, cbar=True, square=True)
    plt.title('Decision Tree Classifier confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("matrix" + str((1-p_test)*100) +
                "_" + str(p_test*100) + ".png")
    return


def calAccuracy(feature_test, label_test):
    label_predict = clf.predict(feature_test)
    print("Accuracy score:", accuracy_score(label_test, label_predict))
    return


def solveRequire4(feature_train, label_train, depth, feature_test, label_test):

    clf = tree.DecisionTreeClassifier(
        criterion="gini", random_state=42, min_samples_leaf=43)
    labelencoder = LabelEncoder()
    for column in df.columns:
        df[column] = labelencoder.fit_transform(df[column])

    if depth == 0:
        clf = DecisionTreeClassifier()
    else:
        clf = DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(feature_train, label_train)

    fig = plt.figure(figsize=(50, 40))
    _ = tree.plot_tree(clf,
                       feature_names=X.columns,
                       filled=True, rounded=True, fontsize=10)
    plt.savefig("tree" + str(depth) + ".png")

    label_predict = clf.predict(feature_test)
    print("Accuracy score in depth:", depth, " ",
          accuracy_score(label_test, label_predict))
    return accuracy_score(label_test, label_predict)


def drawChart(arrAccuracy):
    df1 = pd.DataFrame(arrAccuracy)
    df1.columns = ['Accuracy']
    df = df1.reindex(range(2, 43, 1))
    df.plot()
    plt.title("Cross Validation Accuracy vs Depth of tree")
    plt.xlabel("Depth of tree")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 0.8])
    plt.xlim([2, 42])
    plt.savefig("chart.png")
    return


if __name__ == "__main__":  
    if (len(sys.argv) == 3):
        testsize = float(sys.argv[2])
        split = splitData(testsize) # 0: feature_train, 1: feature_test, 2: label_train, 3: label_test
        clf = buildDecisionTree(split[0], split[2], 0, testsize)
        classificateReport(clf, split[1], split[3])
        confusionMatrix(split[1], split[3], 0.1)
        calAccuracy(split[1], split[3])
    elif (len(sys.argv) == 1):
        split = splitData(0.2)
        arr = []
        for i in range(2, 43):
            arr.append(solveRequire4(
                split[0], split[2], i, split[1], split[3]))
        drawChart(arr)
    else:
        print("Invalid input!!! \n")
