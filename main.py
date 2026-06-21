# This file is used to test the Gemini AI calls to ensure effective running

# Importing the os module and loading the .env file for the API key
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Importing the Gemini AI SDK
from google import genai
client = genai.Client()  # Declare a client

# Declare response
response = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = input('Enter your prompt: ')
)
print(response.text)    #view response