import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, plot_confusion_matrix

df = pd.read_csv("./data/diabetes.csv")


X = df[["BMI", "Age", "Glucose", "BloodPressure"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

# model = LogisticRegression(class_weight = {0: 1, 1: 5})
model = LogisticRegression(class_weight = "balanced")
model.fit(X_train, y_train)

y_test_pred = model.predict_proba(X_test)[:, 1] >= 0.5

precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)

print("Precision: " + str(precision))
print("Recall: " + str(recall))

plot_confusion_matrix(model, X_test, y_test, normalize = "all");