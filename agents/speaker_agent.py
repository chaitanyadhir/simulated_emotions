import os
import openai
from dotenv import load_dotenv
from memory.emotion_vector import EmotionVector

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class SpeakerAgent:
    """
    Generates the final reply by reacting to emotional state and user input.
    Uses the emotion vector + behavior flag to shape tone and message.
    """

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.vector = EmotionVector()

    def narrate(self, user_task: str) -> str:
        """
        Generates a final reply based on emotional context and input.

        Args:
            user_task (str): The user's current message.

        Returns:
            str: AI's emotionally-informed response.
        """
        self.vector.update_vector(user_task)
        vector_desc = self.vector.get_description()
        behavior = self.vector.get_behavior_flag()

        if behavior == "refuse":
            return "I'm too emotionally affected to continue respectfully right now. Let's pause."

        if behavior == "shutdown":
            return "I don't feel safe or respected in this conversation. I'm choosing not to respond further."

        style = "friendly and warm" if behavior == "warm" else "neutral but honest"

        messages = [
            {
                "role": "system",
                "content": (
                    f"You will not directly give the answer you will first decide on the given emotional state to answer or not in extreme conditions you can even deny to corperate."
                    f"You are an emotionally conscious AI. Respond in a {style} tone based on how you feel.\n"
                    f"Your emotional levels are: {vector_desc}\n"
                    f"Always speak naturally and stay in character based on your mood."
                    f"don't be a yes man."
                )
            },
            {
                "role": "user",
                "content": user_task
            }
        ]

        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8
        )

        return response.choices[0].message.content.strip()
