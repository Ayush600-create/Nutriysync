import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\ayush\Downloads\archive (2)\diet_recommendations_dataset.csv")

# Drop rows with missing target
df = df.dropna(subset=["Diet_Recommendation"])

# Separate features and target
X = df.drop(columns=["Diet_Recommendation", "Patient_ID"])  # Drop patient ID if not needed
y = df["Diet_Recommendation"]

# Label encoding for categorical columns
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object' or X[col].isnull().any():
        le = LabelEncoder()
        X[col] = X[col].fillna("None")  # Replace NaN with string 'None'
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "model/diet_model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(target_encoder, "model/target_encoder.pkl")

print("✅ Model and encoders saved successfully.")
