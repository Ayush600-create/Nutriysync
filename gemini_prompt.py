def build_diet_prompt(diet_type, bmi_category, calories, water_liters, preferred_cuisine, dietary_preference):
    """
    Build a detailed prompt for Gemini to generate a 7-day diet plan based on user inputs.
    
    Parameters:
        diet_type (str): Type of diet (e.g., 'Weight Loss', 'Muscle Gain', 'Maintenance')
        bmi_category (str): BMI category (e.g., 'Underweight', 'Normal', 'Overweight', 'Obese')
        calories (int): Recommended daily calorie intake
        water_liters (float): Recommended daily water intake in liters
        preferred_cuisine (str): User's preferred cuisine style
        dietary_preference (str): Dietary preferences/restrictions (e.g., 'Vegetarian', 'Vegan', 'None')
    
    Returns:
        str: A formatted prompt for Gemini AI
    """
    prompt = f"""
    As a nutrition expert, create a detailed and personalized 7-day meal plan with the following specifications:

    USER PROFILE:
    - BMI Category: {bmi_category}
    - Diet Goal: {diet_type}
    - Daily Calorie Target: {calories} calories
    - Daily Water Intake: {water_liters} liters
    - Preferred Cuisine: {preferred_cuisine}
    - Dietary Preference: {dietary_preference}

    REQUIREMENTS:
    1. Create a COMPLETE 7-day meal plan with breakfast, lunch, dinner, and 2 snacks for each day
    2. For EACH meal, include:
       - Meal name
       - Brief description
       - Approximate calories
       - Key nutrients provided
    3. Each day should total approximately {calories} calories
    4. Include hydration reminders throughout the day to reach {water_liters} liters
    5. Focus primarily on {preferred_cuisine} cuisine while incorporating variety
    6. Respect {dietary_preference} dietary preferences in all meal suggestions
    7. For {diet_type} goals, emphasize appropriate macronutrient distribution
    8. Include one weekly grocery list organized by food categories
    
    FORMAT:
    - Use clear headings for each day and meal
    - Present the information in a well-structured, easy-to-follow HTML format
    - Make the plan visually appealing with appropriate spacing and organization
    - Include a brief introduction explaining how this plan addresses the user's specific needs

    Provide ONLY the meal plan without unnecessary explanations or disclaimers.
    """
    
    return prompt
