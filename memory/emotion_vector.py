import os
import json
from typing import Dict
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

VECTOR_FILE = "memory/emotion_vector.json"

# Plutchik pairs: Joy-Sadness, Trust-Disgust, Fear-Anger
PLUTCHIK_AXES = [
    ("joy", "sadness"),
    ("trust", "disgust"),
    ("fear", "anger")
]

class EmotionVector:
    def __init__(self):
        if not os.path.exists("memory"):
            os.makedirs("memory")
        if not os.path.exists(VECTOR_FILE):
            self._write_vector({
                "joy": 0.0,
                "sadness": 0.0,
                "anger": 0.0,
                "fear": 0.0,
                "trust": 0.0,
                "disgust": 0.0
            })

    def _write_vector(self, vector: Dict[str, float]) -> None:
        with open(VECTOR_FILE, "w") as f:
            json.dump(vector, f)

    def _read_vector(self) -> Dict[str, float]:
        with open(VECTOR_FILE, "r") as f:
            return json.load(f)

    def get_vector(self) -> Dict[str, float]:
        return self._read_vector()

    def get_plutchik_position(self) -> Dict[str, float]:
        v = self._read_vector()
        return {
            "joy_sadness": round(v["joy"] - v["sadness"], 4),
            "trust_disgust": round(v["trust"] - v["disgust"], 4),
            "fear_anger": round(v["fear"] - v["anger"], 4)
        }

    def update_vector(self, user_input: str) -> None:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an emotion analysis module. Given any user input, "
                    "you must rate the following 6 emotions from 0.0 to 1.0: "
                    "joy, sadness, anger, fear, trust, and disgust. "
                    "Respond ONLY in valid JSON format as a dictionary with these keys."
                )
            },
            {
                "role": "user",
                "content": user_input
            }
        ]

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )

        try:
            new_vector = json.loads(response.choices[0].message.content.strip())
            current = self._read_vector()

            blended = {
                key: round(min(max((0.7 * current[key] + 0.3 * new_vector[key]), 0.0), 1.0), 4)
                for key in current
            }

            self._write_vector(blended)

        except Exception:
            print(" Failed to parse emotion vector. Keeping last state.")

    def get_description(self) -> str:
        axes = self.get_plutchik_position()
        return (
            f"Plutchik Emotional Axes → Joy-Sadness: {axes['joy_sadness']:.2f}, "
            f"Trust-Disgust: {axes['trust_disgust']:.2f}, Fear-Anger: {axes['fear_anger']:.2f}"
        )

    def get_behavior_flag(self) -> str:
        p = self.get_plutchik_position()
        if p["fear_anger"] < -0.6:
            return "defensive"
        if p["joy_sadness"] > 0.7 and p["trust_disgust"] > 0.5:
            return "engaging"
        if p["joy_sadness"] < -0.6:
            return "withdraw"
        return "neutral"

    def reset(self) -> None:
        self._write_vector({
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "trust": 0.0,
            "disgust": 0.0
        })