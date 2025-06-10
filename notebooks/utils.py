import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import re
from sklearn.preprocessing import LabelEncoder

 #OLD variant:
""" def plot_confusion_matrix(y_test, y_pred, x_labels, y_labels, cmap="Blues"):
    xticklabels = x_labels
    yticklabels = y_labels
    
    # define confusion matrix:
    conf_matrix = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))), dtype=int)
    # count how often each true class /pred class combination occurs:
    for i in range(len(y_test)):
        true_class = y_test[i]
        pred_class = y_pred[i]
        conf_matrix[true_class, pred_class] += 1
    # vizualize the matrix:
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels)
    #sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"Confusion Matrix - Support: {len(y_test)}")
    plt.show() """


def plot_confusion_matrix(y_test, y_pred, x_labels, y_labels, header_text="", cmap="Blues"):
    xticklabels = x_labels
    yticklabels = y_labels

    complete = np.unique(np.concatenate((y_test,y_pred), axis=0))
    le = LabelEncoder()
    le.fit(complete)
    y_test_indices = le.transform(y_test)
    y_pred_indices = le.transform(y_pred)
    
    # define confusion matrix:
    conf_matrix = np.zeros((len(complete), len(complete)), dtype=int)
    # count how often each true class /pred class combination occurs:
    for i in range(len(y_test)):
        true_class = y_test_indices[i]
        pred_class = y_pred_indices[i]
        conf_matrix[true_class, pred_class] += 1
    # vizualize the matrix:
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels)
    #sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    
    if header_text=="":
        plt.title(f"Confusion Matrix - Support: {len(y_test)}")
    else:
        plt.title(f"{header_text} - Support: {len(y_test)}")

    plt.show()