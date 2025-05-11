import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

# Confirm API key loaded
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")

print("✅ API Key loaded:", api_key[:8] + "...")

# Configure Gemini API
genai.configure(api_key=api_key)

# Use the correct available model
try:
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
except Exception as e:
    raise RuntimeError(f"Failed to load Gemini model: {str(e)}")


def generate_7_day_diet(
    bmi_category: str,
    goal: str,
    calorie_goal: float,
    water_liters: float,
    preferred_cuisine: str,
    dietary_preference: str,
    recommended_diet_type: str,
    recommended_foods: list,
    foods_to_avoid: list
) -> str:
    """
    Generates a 7-day meal plan using Gemini based on user profile.
    """
    prompt = f"""
You are a professional dietitian.

Create a personalized 7-day meal plan for a user with the following profile:

- BMI Category: {bmi_category}
- Fitness Goal: {goal}
- Calorie Requirement: {calorie_goal} kcal/day
- Daily Water Intake Goal: {water_liters} liters
- Recommended Diet Type: {recommended_diet_type}
- Dietary Preference: {dietary_preference}
- Preferred Cuisine: {preferred_cuisine}
- Recommended Foods: {', '.join(recommended_foods)}
- Foods to Avoid: {', '.join(foods_to_avoid)}

**Important Instructions**:
- Follow dietary preference strictly (e.g., Vegetarian excludes eggs and meat).
- Avoid all foods listed under "Foods to Avoid".
- Use "Recommended Foods" when building meals.
- Include 3 meals and 1 snack per day.
- Mention portion sizes and meal times.
- Keep meals simple, diverse, and culturally relevant.

Format:
Day 1:
- Breakfast:
- Snack:
- Lunch:
- Dinner:

Repeat for all 7 days.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating meal plan: {str(e)}"


def ask_gemini_question(query: str) -> str:
    """
    Handles Q&A with Gemini assistant.
    """
    try:
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"
