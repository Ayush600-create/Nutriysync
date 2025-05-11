from flask import Flask, render_template, request, jsonify
import os
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv
from gemini_nlp import ask_gemini_question, generate_7_day_diet
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load environment variables - use find_dotenv to locate the file
load_dotenv(find_dotenv())

# Check if API key is loaded
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("WARNING: GOOGLE_API_KEY not found in environment variables")
else:
    print(f"API key loaded successfully: {api_key[:5]}...")

app = Flask(__name__)

# Check for model files in multiple possible locations
possible_model_dirs = [
    "model",
    "models", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    "C:\\Users\\ayush\\Desktop\\project1\\model",
    "C:\\Users\\ayush\\Desktop\\project1\\models"
]

# Initialize variables
MODEL_DIR = None
main_model = None
target_encoder = None

# Search for model files in possible directories
for dir_path in possible_model_dirs:
    main_model_path = os.path.join(dir_path, "diet_type_model.pkl")
    target_encoder_path = os.path.join(dir_path, "target_encoder.pkl")
    
    if os.path.exists(main_model_path) and os.path.exists(target_encoder_path):
        MODEL_DIR = dir_path
        print(f"Found model files in: {MODEL_DIR}")
        try:
            main_model = joblib.load(main_model_path)
            target_encoder = joblib.load(target_encoder_path)
            break
        except Exception as e:
            print(f"Error loading models from {dir_path}: {str(e)}")

# If models aren't found, create simple versions as fallback
if MODEL_DIR is None:
    print("No model directories found. Creating a default models directory...")
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

if main_model is None or target_encoder is None:
    print("Creating fallback models...")
    
    # Create a simple target encoder
    diet_types = [
        "High Protein High Calorie", 
        "Balanced High Calorie",
        "High Protein Moderate Carb", 
        "Balanced Diet",
        "Mediterranean Diet", 
        "Moderate Protein Low Carb",
        "Low Carb Diet", 
        "High Protein Very Low Carb",
        "Low Calorie Low Carb"
    ]
    target_encoder = LabelEncoder()
    target_encoder.fit(diet_types)
    
    # Create a sample dataset for a simple model
    np.random.seed(42)
    X_sample = np.random.rand(100, 9)  # 9 features
    y_sample = np.random.choice(range(len(diet_types)), 100)  # Random diet types
    
    # Train a simple model
    main_model = RandomForestClassifier(n_estimators=10, random_state=42)
    main_model.fit(X_sample, y_sample)
    
    # Save the fallback models
    joblib.dump(main_model, os.path.join(MODEL_DIR, "diet_type_model.pkl"))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.pkl"))
    print(f"Saved fallback models to {MODEL_DIR}")

# Categorize BMI
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Calculate BMR (Basal Metabolic Rate)
def calculate_bmr(weight, height, age, gender):
    # Validate all parameters
    if weight is None or height is None or age is None:
        print(f"Warning: invalid parameters - weight: {weight}, height: {height}, age: {age}")
        # Either return a default value or raise an exception
        raise ValueError("Missing required parameters for BMR calculation")
    
    if not gender:
        print("Warning: gender is None or empty, defaulting to male")
        gender = "male"
        
    if gender.lower() == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161



# Get activity multiplier
def get_activity_multiplier(activity_level):
    activity_map = {
        'Sedentary': 1.2,
        'Light': 1.375,
        'Moderate': 1.55,
        'Active': 1.725,
        'Very Active': 1.9
    }
    return activity_map.get(activity_level, 1.2)



# Normalize fitness goal to match dataset
def normalize_fitness_goal(goal):
    goal = goal.lower()
    if "lose" in goal:
        return "Weight Loss"
    elif "gain" in goal or "muscle" in goal:
        return "Muscle Gain"
    else:
        return "Maintenance"

# Get diet details based on diet type
def get_diet_details(diet_type, bmi_category, fitness_goal):
    """
    Get detailed diet recommendations based on diet type
    """
    diet_details = {
        'description': '',
        'macronutrient_ratio': '',
        'recommended_foods': [],
        'foods_to_avoid': []
    }
    
    # Set diet details based on diet type
    if diet_type == "High Protein High Calorie":
        diet_details['description'] = "A diet focused on muscle building and healthy weight gain with high protein and sufficient calories."
        diet_details['macronutrient_ratio'] = "Protein: 30-35%, Carbs: 45-50%, Fat: 20-25%"
        diet_details['recommended_foods'] = [
            "Lean meats (chicken, turkey, lean beef)",
            "Fatty fish (salmon, tuna)",
            "Eggs and dairy products",
            "Legumes and beans",
            "Whole grains (brown rice, oats)",
            "Nuts and seeds",
            "Healthy oils (olive oil, avocado oil)"
        ]
        diet_details['foods_to_avoid'] = [
            "Highly processed foods",
            "Empty calorie foods",
            "Excessive sugar"
        ]
    
    elif diet_type == "Balanced High Calorie":
        diet_details['description'] = "A balanced diet with higher calories to promote healthy weight gain."
        diet_details['macronutrient_ratio'] = "Protein: 20-25%, Carbs: 50-55%, Fat: 25-30%"
        diet_details['recommended_foods'] = [
            "Whole grains and starches",
            "Healthy fats (avocados, nuts, olive oil)",
            "Lean proteins",
            "Fruits and vegetables",
            "Dairy products",
            "Smoothies and meal replacement shakes"
        ]
        diet_details['foods_to_avoid'] = [
            "Very low-calorie foods",
            "Highly processed foods"
        ]
    
    elif diet_type == "High Protein Moderate Carb":
        diet_details['description'] = "A diet focused on muscle building and maintenance with higher protein content."
        diet_details['macronutrient_ratio'] = "Protein: 30-35%, Carbs: 40-45%, Fat: 20-25%"
        diet_details['recommended_foods'] = [
            "Lean proteins (chicken, turkey, fish)",
            "Eggs and egg whites",
            "Greek yogurt",
            "Quinoa and other whole grains",
            "Legumes and beans",
            "Nuts and seeds",
            "Colorful vegetables"
        ]
        diet_details['foods_to_avoid'] = [
            "Simple sugars",
            "Processed foods",
            "Fried foods",
            "Excessive alcohol"
        ]
    
    elif diet_type == "Balanced Diet":
        diet_details['description'] = "A well-rounded diet with balanced macronutrients for overall health maintenance."
        diet_details['macronutrient_ratio'] = "Protein: 20-25%, Carbs: 45-50%, Fat: 25-30%"
        diet_details['recommended_foods'] = [
            "Varied fruits and vegetables",
            "Whole grains",
            "Lean proteins",
            "Low-fat dairy",
            "Healthy fats (olive oil, avocados, nuts)",
            "Plenty of water"
        ]
        diet_details['foods_to_avoid'] = [
            "Trans fats",
            "Excessive sodium",
            "Added sugars",
            "Highly processed foods"
        ]
    
    elif diet_type == "Mediterranean Diet":
        diet_details['description'] = "A heart-healthy diet based on the eating habits of Mediterranean countries."
        diet_details['macronutrient_ratio'] = "Protein: 15-20%, Carbs: 40-45%, Fat: 35-40% (mostly unsaturated)"
        diet_details['recommended_foods'] = [
            "Olive oil",
            "Nuts and seeds",
            "Fruits and vegetables",
            "Fish and seafood",
            "Whole grains",
            "Legumes",
            "Moderate amounts of dairy, eggs, and poultry",
            "Herbs and spices"
        ]
        diet_details['foods_to_avoid'] = [
            "Red meat (limit consumption)",
            "Processed foods",
            "Added sugars",
            "Refined grains"
        ]
    
    elif diet_type == "Moderate Protein Low Carb":
        diet_details['description'] = "A diet with reduced carbohydrates and moderate protein for active individuals trying to lose weight."
        diet_details['macronutrient_ratio'] = "Protein: 25-30%, Carbs: 25-30%, Fat: 40-45%"
        diet_details['recommended_foods'] = [
            "Lean proteins (chicken, turkey, fish)",
            "Non-starchy vegetables",
            "Healthy fats (avocados, nuts, seeds)",
            "Limited amounts of whole grains",
            "Low-sugar fruits (berries)",
            "Legumes in moderation"
        ]
        diet_details['foods_to_avoid'] = [
            "Refined carbohydrates",
            "Sugary foods and beverages",
            "Processed foods",
            "High-starch foods"
        ]
    
    elif diet_type == "Low Carb Diet":
        diet_details['description'] = "A diet with reduced carbohydrate intake to promote weight loss and improve metabolic health."
        diet_details['macronutrient_ratio'] = "Protein: 25-30%, Carbs: 15-25%, Fat: 45-60%"
        diet_details['recommended_foods'] = [
            "Meat, poultry, and fish",
            "Eggs",
            "Non-starchy vegetables",
            "Nuts and seeds",
            "Healthy oils",
            "Cheese and full-fat dairy",
            "Berries in moderation"
        ]
        diet_details['foods_to_avoid'] = [
            "Sugar and sugary foods",
            "Grains and starches",
            "High-carb fruits",
            "Legumes in large amounts",
            "Highly processed low-carb foods"
        ]
    
    elif diet_type == "High Protein Very Low Carb":
        diet_details['description'] = "A ketogenic-style diet with very low carbohydrates and higher protein for active obese individuals."
        diet_details['macronutrient_ratio'] = "Protein: 30-35%, Carbs: 5-10%, Fat: 55-65%"
        diet_details['recommended_foods'] = [
            "Meats and poultry",
            "Fatty fish",
            "Eggs",
            "Low-carb vegetables",
            "High-fat dairy",
            "Nuts and seeds",
            "Avocados",
            "Healthy oils"
        ]
        diet_details['foods_to_avoid'] = [
            "All sugary foods",
            "Grains and starches",
            "Most fruits",
            "Legumes",
            "Root vegetables",
            "Unhealthy fats"
        ]
    
    elif diet_type == "Low Calorie Low Carb":
        diet_details['description'] = "A calorie-restricted diet with lower carbohydrates for weight loss in obese individuals."
        diet_details['macronutrient_ratio'] = "Protein: 25-30%, Carbs: 20-25%, Fat: 45-50%"
        diet_details['recommended_foods'] = [
            "Lean proteins",
            "Non-starchy vegetables in abundance",
            "Limited fruits (mostly berries)",
            "Limited nuts and seeds",
            "Low-fat dairy products",
            "Healthy oils in moderation"
        ]
        diet_details['foods_to_avoid'] = [
            "Refined carbohydrates",
            "Added sugars",
            "Fried and fast foods",
            "High-calorie condiments",
            "Sugary beverages",
            "Alcohol"
        ]
    
    return diet_details

@app.route('/')
def home():
    return render_template('index.html')




@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        # Get form data
        gender = request.form['gender']
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        activity_level = request.form['activity_level'].title()
        goal = request.form['goal']
        normalized_goal = normalize_fitness_goal(goal)

        medical_conditions = request.form.get('medical_conditions', '')
        preferred_cuisine = request.form.get('preferred_cuisine', 'No Preference')
        dietary_preference = request.form.get('dietary_preference', '')

        # Calculate BMI and determine BMI category
        height_m = height / 100
        bmi = round(weight / (height_m ** 2), 2)
        bmi_category = categorize_bmi(bmi)
        
        # Calculate BMR and maintenance calories
        bmr = calculate_bmr(weight, height, age, gender)
        activity_multiplier = get_activity_multiplier(activity_level)
        maintenance_calories = round(bmr * activity_multiplier)
        
        # Calculate water intake
        base_water = 2.5 if gender.lower() == 'male' else 2.0
        activity_factor = (activity_multiplier - 1.0) * 0.5
        water_liters = round(base_water + activity_factor, 1)
        
        # Calculate calorie goal based on BMI category and fitness goal
        if bmi_category == "Underweight":
            calorie_goal = maintenance_calories + 500  # Surplus for weight gain
        elif bmi_category == "Normal":
            if "muscle" in normalized_goal.lower():
                calorie_goal = maintenance_calories + 300  # Slight surplus for muscle gain
            else:
                calorie_goal = maintenance_calories  # Maintenance
        elif bmi_category == "Overweight":
            calorie_goal = maintenance_calories - 300  # Moderate deficit
        else:  # Obese
            calorie_goal = maintenance_calories - 500  # Larger deficit
        
        # Prepare input data for model
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Weight_kg': [weight],
            'Height_cm': [height],
            'BMI': [bmi],
            'BMI_Category': [bmi_category],
            'Activity_Level': [activity_level],
            'Fitness_Goal': [normalized_goal],
            'Maintenance_Calories': [maintenance_calories]
        })
        
        # Choose appropriate model based on BMI category
        specific_model_path = os.path.join(MODEL_DIR, f"diet_type_{bmi_category.lower()}_model.pkl")
        
        try:
            if os.path.exists(specific_model_path):
                specific_model = joblib.load(specific_model_path)
                prediction = specific_model.predict(input_data)
            else:
                prediction = main_model.predict(input_data)
                
            recommended_diet_type = target_encoder.inverse_transform(prediction)[0]
        except Exception as e:
            print(f"Error predicting diet type: {str(e)}")
            # Fallback recommendation based on BMI category and goal
            if bmi_category == "Underweight":
                recommended_diet_type = "Balanced High Calorie"
            elif bmi_category == "Normal":
                if normalized_goal == "Muscle Gain":
                    recommended_diet_type = "High Protein Moderate Carb"
                else:
                    recommended_diet_type = "Balanced Diet"
            elif bmi_category == "Overweight":
                recommended_diet_type = "Moderate Protein Low Carb"
            else:  # Obese
                recommended_diet_type = "Low Calorie Low Carb"
        
        # Get detailed diet recommendations
        diet_details = get_diet_details(recommended_diet_type, bmi_category, normalized_goal)
        
        # More detailed debugging
        print(f"About to generate meal plan with diet type: {recommended_diet_type}")
        print(f"Calorie goal: {calorie_goal}, BMI category: {bmi_category}")
        
        # Check for API key before attempting to generate meal plan
        if not api_key:
            print("WARNING: TOGETHER_API_KEY not found for meal plan generation")
            seven_day_plan = (
                "<div class='alert alert-danger'>"
                "<h4>Unable to Generate Meal Plan</h4>"
                "<p>We encountered an error with our AI service due to missing API key configuration.</p>"
                "<p>Please make sure your GOOGLE_API_KEY is correctly set in your .env file.</p>"
                "</div>"
            )
        else:
            # Generate 7-day meal plan using Gemini NLP
            try:
                print("Calling generate_7_day_diet function...")
                raw_plan = generate_7_day_diet(
                    bmi_category,
                    normalized_goal,
                    calorie_goal,
                    water_liters,
                    preferred_cuisine,
                    dietary_preference,
                    recommended_diet_type,
                    diet_details['recommended_foods'],
                    diet_details['foods_to_avoid']
                )
                
                print(f"Generated plan response type: {type(raw_plan)}")
                if raw_plan:
                    print(f"First 100 chars of plan: {raw_plan[:100] if len(raw_plan) > 100 else raw_plan}")
                
                # Check if the response starts with "Error:" (from our error handling)
                if raw_plan and raw_plan.startswith("Error:"):
                    raise Exception(raw_plan)
                
                # Prepare the result
                pref_note = (
                    f"🌿 <strong>This is your 7-day diet plan based on your "
                    f"<span style='color:green;'>{preferred_cuisine}</span> cuisine preference and "
                    f"recommended <span style='color:green;'>{recommended_diet_type}</span> diet.</strong><br><br>"
                )
                seven_day_plan = pref_note + "<pre>" + raw_plan + "</pre>"
            except Exception as e:
                error_message = str(e)
                print(f"Error generating meal plan: {error_message}")
                
                # Create a user-friendly error message
                if "API key" in error_message.lower():
                    seven_day_plan = (
                        "<div class='alert alert-danger'>"
                        "<h4>Unable to Generate Meal Plan</h4>"
                        "<p>We encountered an error with our AI service due to an API key configuration issue.</p>"
                        "<p>Please make sure your GOOGLE_API_KEY is correctly set in your .env file.</p>"
                        "</div>"
                    )
                elif "quota" in error_message.lower():
                    seven_day_plan = (
                        "<div class='alert alert-warning'>"
                        "<h4>Unable to Generate Meal Plan</h4>"
                        "<p>We've reached our API quota limit. Please try again later.</p>"
                        "</div>"
                    )
                elif "safety" in error_message.lower() or "blocked" in error_message.lower():
                    seven_day_plan = (
                        "<div class='alert alert-warning'>"
                        "<h4>Unable to Generate Meal Plan</h4>"
                        "<p>The request was blocked by content safety filters. We're working to adjust our prompts.</p>"
                        "</div>"
                    )
                else:
                    seven_day_plan = (
                        "<div class='alert alert-warning'>"
                        "<h4>Unable to Generate Meal Plan</h4>"
                        "<p>We encountered an error while generating your personalized meal plan.</p>"
                        f"<p>Error details: {error_message}</p>"
                        "<p>Please try again later or contact support if the issue persists.</p>"
                        "</div>"
                    )

        return render_template(
            'result.html',
            bmi=bmi,
            bmi_category=bmi_category,
            maintenance_calories=maintenance_calories,
            calorie_goal=calorie_goal,
            water_liters=water_liters,
            recommended_diet_type=recommended_diet_type,
            diet_description=diet_details['description'],
            macronutrient_ratio=diet_details['macronutrient_ratio'],
            recommended_foods=diet_details['recommended_foods'],
            foods_to_avoid=diet_details['foods_to_avoid'],
            seven_day_plan=seven_day_plan
        )

    except Exception as e:
        error_message = str(e)
        print(f"Critical error in get_recommendations route: {error_message}")
        return render_template(
            'error.html',
            error_message=f"An error occurred while processing your request: {error_message}"
        )

@app.route('/ask-assistant', methods=['POST'])
def ask_assistant():
    user_query = request.form.get('user_query')
    if not user_query:
        return "Please enter a query."
    
    print(f"Processing user query: {user_query[:50]}...")
    
    # Check for API key
    if not api_key:
        return render_template('result.html',
            bmi=None,
            bmi_category=None,
            maintenance_calories=None,
            water_liters=None,
            recommended_diet_type=None,
            seven_day_plan=None,
            nlp_diet="<div class='alert alert-danger'><h4>API Key Error</h4><p>Google API Key is missing. Please set up your GOOGLE_API_KEY in the .env file.</p></div>"
        )
    
    try:
        print("Calling ask_gemini_question function...")
        response = ask_gemini_question(user_query)
        print(f"Received response length: {len(response) if response else 0}")
        
        # Check if response is an error message
        if response and response.startswith("Error:"):
            error_message = response[6:] # Remove "Error: " prefix
            print(f"Error from Gemini API: {error_message}")
            
            if "quota" in response.lower():
                response = f"<div class='alert alert-warning'><h4>API Quota Exceeded</h4><p>We've reached our daily limit for AI queries. Please try again tomorrow.</p></div>"
            elif "rate" in response.lower():
                response = f"<div class='alert alert-warning'><h4>Too Many Requests</h4><p>We're processing too many requests right now. Please try again in a few minutes.</p></div>"
            else:
                response = f"<div class='alert alert-warning'><h4>Unable to process your question</h4><p>{error_message}</p></div>"
        else:
            # Format the successful response
            response = f"<div class='diet-assistant-response'>{response}</div>"
            
    except Exception as e:
        error_message = str(e)
        print(f"Exception in ask_assistant: {error_message}")
        response = f"<div class='alert alert-danger'><h4>Error</h4><p>{error_message}</p></div>"
    
    return render_template('result.html',
        bmi=None,
        bmi_category=None,
        maintenance_calories=None,
        water_liters=None,
        recommended_diet_type=None,
        seven_day_plan=None,
        nlp_diet=response
    )

@app.route('/api/meal-plan-json', methods=['POST'])
def get_meal_plan_json():
    """
    Endpoint to generate a structured meal plan in JSON format.
    This is used by the advanced frontend visualization.
    """
    try:
        gender = request.form.get('gender', 'male')
        age = int(request.form.get('age', 30))
        height = float(request.form.get('height', 170))
        weight = float(request.form.get('weight', 70))
        activity_level = request.form.get('activity_level', 'Moderate')
        goal = request.form.get('goal', 'Maintenance')
        normalized_goal = normalize_fitness_goal(goal)
        preferred_cuisine = request.form.get('preferred_cuisine', 'No Preference')
        dietary_preference = request.form.get('dietary_preference', '')

        height_m = height / 100
        bmi = round(weight / (height_m ** 2), 2)
        bmi_category = categorize_bmi(bmi)

        bmr = calculate_bmr(weight, height, age, gender)
        activity_multiplier = get_activity_multiplier(activity_level)
        maintenance_calories = round(bmr * activity_multiplier)

        if bmi_category == "Underweight":
            calorie_goal = maintenance_calories + 500
        elif bmi_category == "Normal":
            if "muscle" in normalized_goal.lower():
                calorie_goal = maintenance_calories + 300
            else:
                calorie_goal = maintenance_calories
        elif bmi_category == "Overweight":
            calorie_goal = maintenance_calories - 300
        else:  # Obese
            calorie_goal = maintenance_calories - 500

        if bmi_category == "Underweight":
            recommended_diet_type = "Balanced High Calorie"
        elif bmi_category == "Normal":
            if normalized_goal == "Muscle Gain":
                recommended_diet_type = "High Protein Moderate Carb"
            else:
                recommended_diet_type = "Balanced Diet"
        elif bmi_category == "Overweight":
            recommended_diet_type = "Moderate Protein Low Carb"
        else:
            recommended_diet_type = "Low Calorie Low Carb"

        base_water = 2.5 if gender.lower() == 'male' else 2.0
        activity_factor = (activity_multiplier - 1.0) * 0.5
        water_liters = round(base_water + activity_factor, 1)

        if not api_key:
            return jsonify({
                "error": "API key not configured. Please set up GOOGLE_API_KEY in your .env file."
            })

        # Fetch diet details
        diet_details = get_diet_details(recommended_diet_type, bmi_category, normalized_goal)

        # Generate meal plan using Gemini
        raw_plan = generate_7_day_diet(
            bmi_category,
            normalized_goal,
            calorie_goal,
            water_liters,
            preferred_cuisine,
            dietary_preference,
            recommended_diet_type,
            diet_details['recommended_foods'],
            diet_details['foods_to_avoid']
        )

        if raw_plan.startswith("Error:"):
            return jsonify({"error": raw_plan})

        return jsonify({
            "bmi": bmi,
            "bmi_category": bmi_category,
            "maintenance_calories": maintenance_calories,
            "calorie_goal": calorie_goal,
            "water_liters": water_liters,
            "recommended_diet_type": recommended_diet_type,
            "preferred_cuisine": preferred_cuisine,
            "dietary_preference": dietary_preference,
            "macronutrient_ratio": diet_details['macronutrient_ratio'],
            "recommended_foods": diet_details['recommended_foods'],
            "foods_to_avoid": diet_details['foods_to_avoid'],
            "seven_day_plan": raw_plan
        })

    except Exception as e:
        error_message = str(e)
        print(f"Error in get_meal_plan_json: {error_message}")
        return jsonify({
            "error": f"An error occurred: {error_message}"
        })


@app.route('/api/ask-nutrina', methods=['POST'])
def api_ask_nutrina():
    """
    API endpoint for the chat assistant functionality
    """
    try:
        user_query = request.json.get('query', '')
        if not user_query:
            return jsonify({"error": "Please provide a query"})
        
        if not api_key:
            return jsonify({"error": "API key not configured"})
            
        response = ask_gemini_question(user_query)
        
        # Check if response is an error message
        if response and response.startswith("Error:"):
            error_message = response[6:] # Remove "Error: " prefix
            return jsonify({"error": error_message})
            
        return jsonify({"response": response})
        
    except Exception as e:
        return jsonify({"error": str(e)})

# Add a fallback error handler
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    print(f"Unhandled exception: {str(e)}")
    # Return a friendly error page
    return render_template('error.html', error_message=str(e)), 500

# Add a route for errors
@app.route('/error')
def error():
    return render_template('error.html', error_message="Something went wrong. Please try again.")

import webbrowser

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)


