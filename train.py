
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Read in data
X_train = np.genfromtxt("data/train_features.csv", delimiter=',')
y_train = np.genfromtxt("data/train_labels.csv", delimiter=',')
X_test = np.genfromtxt("data/test_features.csv", delimiter=',')
y_test = np.genfromtxt("data/test_labels.csv", delimiter=',')

# Fit a model
depth = 5
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train, y_train)

# Get accuracy
acc = clf.score(X_test, y_test)
print(f"Accuracy: {acc:.4f}")

# Write accuracy to metrics file
metrics = f"""
Accuracy: {acc:.4f}

![Confusion Matrix](plot.png)
"""
with open("metrics.txt", "w") as outfile:
    outfile.write(metrics)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues)
plt.savefig("plot.png")
