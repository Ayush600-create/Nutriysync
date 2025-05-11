# Nutriysync
NutriSync is a personalized diet and wellness web application built with Flask and powered by machine learning and AI integration. It helps users track their health metrics and provides tailored diet plans based on their individual profiles.

#Key Features

BMI & BMR Calculator – Calculates your Body Mass Index and Basal Metabolic Rate using user input
Daily Water Intake Estimator – Recommends ideal daily hydration levels
Maintenance Calorie & Goal Setting – Calculates daily calorie needs and weight goals
AI-Powered Meal Planning – Uses the Gemini API to generate 7-day personalized vegan Indian meal plans
Fallback Meal Logic – If the API quota is exceeded, rule-based logic ensures continued functionality
Machine Learning Diet Type Prediction – Predicts suitable diet types (e.g., low-carb, high-protein) based on user profile using a trained ML model
User-Friendly UI – Organized with tabbed navigation and clear results

#Tech Stack

Backend: Python, Flask
Frontend: HTML, CSS (Jinja templates)
Database: SQLite
ML Model: RandomForestClassifier for diet prediction
External API: Google Gemini for natural language-based meal planning
Environment Management: dotenv

#Getting Started

-Prerequisites

Python 3.8 or higher
pip package manager
Google Gemini API key (for AI-powered meal planning)

#Installation

Clone the repository
bashgit clone https://github.com/yourusername/nutrisync.git
cd nutrisync

Create and activate a virtual environment
bashpython -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

Install dependencies
bashpip install -r requirements.txt

Set up environment variables
bash# Create a .env file in the project root
touch .env

# Add the following to the .env file
GEMINI_API_KEY=your_gemini_api_key

Initialize the database
bashpython init_db.py

Run the application
bashpython app.py

Access the application at http://localhost:5000

#Usage

Enter your health metrics (age, height, weight, etc.)
Set your goals (maintain, lose, or gain weight)
Get your personalized diet plan based on your profile
Use the tabbed navigation to access different features of the application

📊 Project Structure
nutrisync/
├── app.py                    # Main Flask application
├── models.py                 # ML models and data structures
├── predict.py                # Diet prediction functionality
├── train_model.py            # Training script for ML model
├── retrain_model.py          # Model retraining functionality
├── test_model.py             # Testing script for ML model
├── gemini_nlp.py             # Gemini API integration for NLP
├── gemini_prompt.py          # Prompt templates for Gemini API
├── test_gemini.py            # Testing script for Gemini integration
├── .env                      # Environment variables (not tracked in git)
├── bmi.db                    # Database file
├── health.db                 # Health metrics database
├── diet_planner.log          # Application logs
├── _pycache_/                # Python cache directory
├── instance/                 # Flask instance folder
├── meal_plan_cache/          # Cached meal plans
├── model/                    # Saved ML model files
├── models/                   # Model definitions
├── venv/                     # Virtual environment
└── templates/                # Jinja templates
    ├── index.html            # Main application page
    ├── result.html           # Results display page
    ├── recommendation.html   # Diet recommendations page
    └── error.html            # Error handling page
🧠 Machine Learning Model
The diet prediction model uses a RandomForestClassifier trained on user profiles and their optimal diet types. It takes into account:

Age
Gender
Height
Weight
Activity level
Health conditions
Goals

Based on these inputs, it recommends one of the following diet types:

Low-carb
High-protein
Balanced
Plant-based
Mediterranean

🤖 AI Integration
The meal planning feature uses Google's Gemini API to generate personalized meal plans. It takes into account:

Dietary restrictions
Allergies
Cultural preferences
Nutritional goals

If the API quota is exceeded, the application falls back to a rule-based system that selects meals from a pre-defined database.
🔒 Security Features

Form validation
Input sanitization
Environment variable management for API keys

🔄 Future Enhancements

User authentication system
Mobile app version
Integration with fitness trackers
Community features
Grocery list generation
Recipe customization
Progress tracking over time


#Contributors

Ayush Sharma

#Acknowledgements

Flask Documentation
Google Gemini API
scikit-learn
Python
Jinja2
SQLite
