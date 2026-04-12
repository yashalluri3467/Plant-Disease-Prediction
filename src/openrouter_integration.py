import os
import requests
from dotenv import load_dotenv
from config import OPENROUTER_MODEL, OPENROUTER_TIMEOUT_SECONDS

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"

def get_disease_advice(disease_name, crop_name=None):
    if not API_KEY or API_KEY == "your_openrouter_api_key_here":
        return "OpenRouter API key is not configured."

    crop_context = f" for {crop_name}" if crop_name else ""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Provide practical treatment and prevention guidance for {disease_name}{crop_context}. "
                    "Keep it concise with: immediate remedy steps, prevention tips, and when to seek expert help."
                )
            }
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(
            URL, headers=headers, json=payload, timeout=OPENROUTER_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException:
        return "Unable to fetch AI advice at the moment."
