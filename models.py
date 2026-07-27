import os
from openai import OpenAI

client = OpenAI(api_key="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")

def get_gpt_prediction(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""You are analyzing text for signs of emotional distress.
Classify the following text as exactly one of these three categories:
High Risk, Medium Risk, or Low Risk

Text: {text}

Respond with only the classification label. Nothing else."""
        }],
        max_tokens=10,
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Test it
if __name__ == "__main__":
    test_sentences = [
        "I had a great day today.",
        "I don't know if anything matters anymore.",
        "I feel tired but okay.",
        "I can't stop crying and I don't know why.",
        "Looking forward to the weekend."
    ]
    
    for sentence in test_sentences:
        result = get_gpt_prediction(sentence)
        print(f"Text: {sentence}")
        print(f"Prediction: {result}")
        print("-" * 50)