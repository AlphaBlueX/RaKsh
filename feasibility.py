# borewell_prediction.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("bengaluru_borewell_dataset.csv")

# Drop rows with missing target values
df = df.dropna(subset=["groundwater_level", "feasibility"])

# Fill missing numeric features with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill missing categorical features with mode
categorical_cols = ['soil_type', 'land_use', 'water_retention', 'aquifer_type', 'water_quality']
for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# Label encode categorical columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ------------------- Feature Preparation -------------------

# Features for both classification & regression
features = ['latitude', 'longitude', 'elevation', 'population_density',
            'soil_type', 'land_use', 'rainfall', 'water_retention',
            'aquifer_type', 'water_quality']

# ------------------- Classification (Feasibility) -------------------
X_cls = df[features]
y_cls = df['feasibility']

# Scale features
scaler_cls = StandardScaler()
X_cls_scaled = scaler_cls.fit_transform(X_cls)

# Train-test split
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls_scaled, y_cls, test_size=0.2, random_state=42)

print("\n---- CLASSIFICATION RESULTS (Feasibility) ----")

# Support Vector Classifier
svc = SVC()
svc.fit(X_train_cls, y_train_cls)
pred_svc = svc.predict(X_test_cls)
print("\nSVC Accuracy:", accuracy_score(y_test_cls, pred_svc))
print(classification_report(y_test_cls, pred_svc))

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train_cls, y_train_cls)
pred_knn = knn.predict(X_test_cls)
print("\nKNN Accuracy:", accuracy_score(y_test_cls, pred_knn))
print(classification_report(y_test_cls, pred_knn))

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train_cls, y_train_cls)
pred_rf = rf.predict(X_test_cls)
print("\nRandom Forest Accuracy:", accuracy_score(y_test_cls, pred_rf))
print(classification_report(y_test_cls, pred_rf))

# Gradient Boosting Classifier
gb = GradientBoostingClassifier()
gb.fit(X_train_cls, y_train_cls)
pred_gb = gb.predict(X_test_cls)
print("\nGradient Boosting Accuracy:", accuracy_score(y_test_cls, pred_gb))
print(classification_report(y_test_cls, pred_gb))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test_cls, pred_rf), annot=True, cmap="Blues", fmt='d')
plt.title("Confusion Matrix - Random Forest (Feasibility)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ------------------- Regression (Groundwater Level) -------------------
X_reg = df[features]
y_reg = df['groundwater_level']

# Scale features
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

# Train-test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

print("\n---- REGRESSION RESULTS (Groundwater Level) ----")

# Support Vector Regression
svr = SVR(kernel='rbf')
svr.fit(X_train_reg, y_train_reg)
pred_svr = svr.predict(X_test_reg)
print("\nSVR R^2 Score:", r2_score(y_test_reg, pred_svr))

# Plot predictions
plt.figure(figsize=(8, 5))
plt.scatter(y_test_reg, pred_svr, alpha=0.6)
plt.xlabel("Actual Groundwater Level")
plt.ylabel("Predicted Groundwater Level")
plt.title("SVR Predictions vs Actual")
plt.grid(True)
plt.tight_layout()
plt.show()
