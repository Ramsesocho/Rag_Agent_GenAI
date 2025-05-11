import openai
import base64
from typing import List
import os
from dotenv import load_dotenv

# Charger la clé OpenAI depuis .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_installation_images(image_paths: List[str], prompt: str) -> str:
    image_contents = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image_to_base64(path)}"
            }
        }
        for path in image_paths
    ]

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}] + image_contents
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=1024
    )

    return response["choices"][0]["message"]["content"]
