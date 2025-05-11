import joblib
import sqlite3
import numpy as np

# Load the model and encoders
model = joblib.load("model/diet_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

def predict_diet(user_id):
    # Connect to the database
    conn = sqlite3.connect("health.db")
    cursor = conn.cursor()

    # Fetch user data
    cursor.execute("""
        SELECT age, gender, height, weight, activity_level, goal, allergies, medical_conditions
        FROM users WHERE id=?
    """, (user_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return "User data not found."

    # Map columns to names
    columns = ['age', 'gender', 'height', 'weight', 'activity_level', 'goal', 'allergies', 'medical_conditions']
    user_data = dict(zip(columns, row))

    # Encode categorical features safely
    for col in ['gender', 'activity_level', 'goal', 'allergies', 'medical_conditions']:
        le = label_encoders[col]
        try:
            user_data[col] = le.transform([user_data[col]])[0]
        except ValueError:
            return f"Value '{user_data[col]}' for '{col}' is not recognized by the model."

    # Prepare feature array
    features = np.array([[ 
        user_data['age'],
        user_data['gender'],
        user_data['height'],
        user_data['weight'],
        user_data['activity_level'],
        user_data['goal'],
        user_data['allergies'],
        user_data['medical_conditions']
    ]])

    # Predict and decode the result
    try:
        prediction_encoded = model.predict(features)
        prediction = target_encoder.inverse_transform(prediction_encoded)[0]
        return prediction
    except Exception as e:
        return f"Error during prediction: {str(e)}"
