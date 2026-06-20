# ==================================================
# LLM Explanation Utility
# ==================================================

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==================================================
# Generate Explanation
# ==================================================

def generate_llm_explanation(prediction: str,confidence: float) -> str:

    prompt = f"""
You are an AI dermatology education assistant.

A multimodal skin lesion classification model produced:

Prediction: {prediction}
Confidence Score: {confidence:.2f}%

Instructions:

1. Explain the prediction in simple language.
2. Provide general skin monitoring and skin care advice.
3. Explain when consultation with a dermatologist may be appropriate.
4. Do NOT provide treatment recommendations.
5. Do NOT prescribe medications.
6. Do NOT claim the user has a disease.
7. Clearly state that this is an AI-generated research prediction and not a medical diagnosis.
8. Keep the response under 120 words.
9. Write as one short paragraph.
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content":
                    (
                        "You are a helpful dermatology "
                        "education assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return (response.choices[0].message.content.strip())

    except Exception as e:

        print(f"LLM Error: {e}")

        return (
            "AI explanation is currently unavailable."
        )