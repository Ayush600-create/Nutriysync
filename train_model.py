import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Step 1: Load dataset
DATA_PATH = r"C:\Users\ayush\Downloads\final_health_dataset_5000.csv"
df = pd.read_csv(DATA_PATH)

# Step 2: Feature Engineering
# Calculate BMI
df['Height_m'] = df['Height_cm'] / 100
df['BMI'] = df['Weight_kg'] / (df['Height_m'] ** 2)

# Categorize BMI
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 24.9:
        return "Normal"
    elif bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"
        
df['BMI_Category'] = df['BMI'].apply(bmi_category)

# Calculate BMR
def calculate_bmr(row):
    if row['Gender'].lower() == 'male':
        return 10 * row['Weight_kg'] + 6.25 * row['Height_cm'] - 5 * row['Age'] + 5
    else:
        return 10 * row['Weight_kg'] + 6.25 * row['Height_cm'] - 5 * row['Age'] - 161
        
df['BMR'] = df.apply(calculate_bmr, axis=1)

# Calculate maintenance calories
activity_map = {
    'Sedentary': 1.2,
    'Light': 1.375,
    'Moderate': 1.55,
    'Active': 1.725,
    'Very Active': 1.9
}
df['Activity_Multiplier'] = df['Activity_Level'].map(activity_map).fillna(1.2)
df['Maintenance_Calories'] = df['Activity_Multiplier'] * df['BMR']

# Step 3: Diet Type Assignment (if not already in dataset)
if 'Diet_Type' not in df.columns:
    def assign_diet_type(row):
        bmi_cat = row['BMI_Category']
        fitness_goal = str(row['Fitness_Goal']).lower() if isinstance(row['Fitness_Goal'], str) else ""
        activity = str(row['Activity_Level']).lower() if isinstance(row['Activity_Level'], str) else ""
        
        # For Underweight individuals
        if bmi_cat == "Underweight":
            if "muscle" in fitness_goal or "strength" in fitness_goal:
                return "High Protein High Calorie"
            else:
                return "Balanced High Calorie"
                
        # For Normal weight individuals
        elif bmi_cat == "Normal":
            if "muscle" in fitness_goal:
                return "High Protein Moderate Carb"
            elif "maintenance" in fitness_goal:
                return "Balanced Diet"
            else:
                return "Mediterranean Diet"
                
        # For Overweight individuals
        elif bmi_cat == "Overweight":
            if "active" in activity or "very active" in activity:
                return "Moderate Protein Low Carb"
            else:
                return "Low Carb Diet"
                
        # For Obese individuals
        else:  # Obese
            if "active" in activity or "very active" in activity:
                return "High Protein Very Low Carb"
            else:
                return "Low Calorie Low Carb"
    
    # Apply the function to create diet type recommendations
    df['Diet_Type'] = df.apply(assign_diet_type, axis=1)

# Print the distribution of diet types
print("Diet Type Distribution:")
print(df['Diet_Type'].value_counts())

# Feature selection
features = [
    'Age', 'Gender', 'Weight_kg', 'Height_cm', 'BMI', 'BMI_Category',
    'Activity_Level', 'Fitness_Goal', 'Maintenance_Calories'
]

# Add additional features if they exist
if 'Preferred_Diet' in df.columns:
    features.append('Preferred_Diet')
if 'Preferred_Cuisine' in df.columns:
    features.append('Preferred_Cuisine')
if 'Health_Issues' in df.columns:
    features.append('Health_Issues')

# Our target is now Diet_Type
target = 'Diet_Type'
X = df[features].copy()
y = df[target]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create preprocessing pipelines
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Encode the target variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Print the target class mapping
print("\nDiet Type Class Mapping:")
for i, diet_type in enumerate(target_encoder.classes_):
    print(f"{i}: {diet_type}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=150, 
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Convert encoded predictions back to original labels for better readability
y_test_labels = target_encoder.inverse_transform(y_test)
y_pred_labels = target_encoder.inverse_transform(y_pred)

# Print counts of actual vs predicted
print("\nActual Diet Type Counts:")
print(pd.Series(y_test_labels).value_counts())
print("\nPredicted Diet Type Counts:")
print(pd.Series(y_pred_labels).value_counts())

# Create specialized models for each BMI category
bmi_categories = ["Underweight", "Normal", "Overweight", "Obese"]
specialized_models = {}

for category in bmi_categories:
    # Filter dataset for specific BMI category
    category_df = df[df['BMI_Category'] == category]
    
    if len(category_df) > 50:  # Only proceed if we have enough samples
        print(f"\nTraining specialized model for {category} individuals")
        print(f"Number of samples: {len(category_df)}")
        
        # Prepare features and target
        X_cat = category_df[features].copy()
        y_cat = category_df[target]
        y_cat_encoded = target_encoder.transform(y_cat)
        
        # Print class distribution
        print(f"Diet Type distribution for {category} individuals:")
        print(category_df[target].value_counts())
        
        # Split data
        X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(
            X_cat, y_cat_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cat_encoded
        )
        
        # Train specialized model
        cat_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=150,
                max_depth=None,
                min_samples_split=2,
                class_weight='balanced',
                random_state=RANDOM_STATE
            ))
        ])
        
        cat_pipeline.fit(X_cat_train, y_cat_train)
        
        # Evaluate specialized model
        y_cat_pred = cat_pipeline.predict(X_cat_test)
        print(f"\nClassification Report for {category} Individuals:")
        print(classification_report(y_cat_test, y_cat_pred))
        
        # Save the specialized model
        specialized_models[category] = cat_pipeline
    else:
        print(f"\nNot enough samples for {category} individuals. Skipping specialized model.")

# Save models and encoders
os.makedirs("../models", exist_ok=True)

# Save main model
joblib.dump(pipeline, "../models/diet_type_model.pkl")

# Save target encoder
joblib.dump(target_encoder, "../models/target_encoder.pkl")

# Save the diet types list
with open("../models/diet_types.txt", "w") as f:
    for diet_type in target_encoder.classes_:
        f.write(f"{diet_type}\n")

# Save specialized models
for category, model in specialized_models.items():
    joblib.dump(model, f"../models/diet_type_{category.lower()}_model.pkl")

print("\nModels and encoders successfully saved in ../models/ directory")
