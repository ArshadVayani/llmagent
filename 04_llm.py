
from openai import OpenAI

# Initialize OpenAI Client (Adjust base_url for local Ollama if needed)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Function to interact with the chat model
def chat_with_model(prompt):
    response = client.completions.create(
        model="smollm2:135m",
        prompt=prompt,
        max_tokens=1000
    )
    return response.choices[0].text.strip()

# Example usage
if __name__ == "__main__":
    user_prompt = "Hello, how can you help me today?"
    response = chat_with_model(user_prompt)
    print("Model Response:", response)
