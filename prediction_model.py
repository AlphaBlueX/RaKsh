import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# ----------- CONFIGURATION -----------
TARGET_COL = 'feasibility'
MODEL_FILENAME = 'borewell_model.pkl'
DATA_FILE = 'bengaluru_borewell_dataset.csv'
# --------------------------------------

# Load dataset
data = pd.read_csv(DATA_FILE)

# Confirm target is binary (already 0 and 1)
if not set(data[TARGET_COL].unique()).issubset({0, 1}):
    raise ValueError(f"The target column '{TARGET_COL}' must contain only 0 or 1.")

# Encode categorical columns (if any)
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != TARGET_COL]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features and target
X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, MODEL_FILENAME)
print("Model trained and saved successfully.\n")

# ----------- User Input and Prediction -----------
try:
    lat = float(input("Enter latitude: ").strip())
    lon = float(input("Enter longitude: ").strip())
except ValueError:
    print("Invalid input. Please enter numeric values for latitude and longitude.")
    exit()
# Use nearest row based on latitude and longitude
feature_cols = data.drop(columns=[TARGET_COL])
distances = ((feature_cols['latitude'] - lat)**2 + (feature_cols['longitude'] - lon)**2).pow(0.5)
nearest_index = distances.idxmin()

input_features = feature_cols.iloc[nearest_index]
prediction = clf.predict([input_features])[0]
result = "Borewell can be dug." if prediction == 1 else "Borewell NOT recommended."
print(f"\nClosest match at index {nearest_index} with coordinates:")
print(f"Latitude: {input_features['latitude']}, Longitude: {input_features['longitude']}")
print(f"Prediction result: {result}")
