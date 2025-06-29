from agents.speaker_agent import SpeakerAgent
from memory.emotion_vector import EmotionVector

def run_simulated_conscious_ai(user_task: str) -> str:
    speaker = SpeakerAgent()
    final_reply = speaker.narrate(user_task)
    return final_reply

if __name__ == "__main__":
    print("Simulated Conscious AI Prototype (Emotion + Speaker)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter your task: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = run_simulated_conscious_ai(user_input)
        print("\n--- FINAL ANSWER ---")
        print(result)

        # Show current emotional state
        vector = EmotionVector()
        print("\n" + vector.get_description())
        print("\n" + "-" * 60 + "\n")
