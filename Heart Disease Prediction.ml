import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load Dataset
df = pd.read_csv("/content/heart.csv")

print(df.head())
print(df.shape)

# Basic Statistics
print(df.describe())

# Check Missing Values
print(df.isnull().sum())

# Correlation Analysis
correlation = df.corr()

plt.figure(figsize=(10,8))
plt.imshow(correlation, cmap="coolwarm")
plt.colorbar()
plt.title("Correlation Matrix")
plt.show()

# Features and Target
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy")
print(lr_accuracy)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy")
print(rf_accuracy)

# Model Comparison
models = ["Logistic Regression", "Random Forest"]
scores = [lr_accuracy * 100, rf_accuracy * 100]

plt.figure(figsize=(8,5))
plt.bar(models, scores)
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.show()

# Feature Importance
importance = rf_model.feature_importances_

feature_names = X.columns

plt.figure(figsize=(10,6))
plt.barh(feature_names, importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

# Best Model
if rf_accuracy > lr_accuracy:
    best_model = "Random Forest"
    best_accuracy = rf_accuracy
else:
    best_model = "Logistic Regression"
    best_accuracy = lr_accuracy

print(f"\nBest Model: {best_model}")
print(f"Accuracy: {best_accuracy*100:.2f}%")

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))