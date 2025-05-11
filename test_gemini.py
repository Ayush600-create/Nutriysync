import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Try with the correct model name
model = genai.GenerativeModel("gemini-1.5-pro")

# Generate a test response
response = model.generate_content("Say hello from Gemini!")

# Print the result
print(response.text)
