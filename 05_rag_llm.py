import faiss
import numpy as np
from openai import OpenAI

# Initialize OpenAI Client (Adjust base_url for local Ollama if needed)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Load FAISS index and policy texts
index = faiss.read_index("policy_index.faiss")
with open("policy_texts.txt", "r") as f:
    policy_texts = f.readlines()

# Function to get embedding for a query
def get_embedding(text):
    response = client.embeddings.create(
        model="all-minilm",
        input=text
    )
    return np.array(response.data[0].embedding).astype('float32')

# Function to perform vector search
def vector_search(query, top_k=4):
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [policy_texts[idx] for idx in indices[0]]

# Function to interact with the chat model
def chat_with_model(prompt):
    response = client.completions.create(
        model="smollm2:360m", # Adjust model name as needed smollm2:360m, llama3.2:1b
        prompt=prompt,
        max_tokens=1000
    )
    return response.choices[0].text.strip()

# Example usage
if __name__ == "__main__":
    user_query = "What is the shipping policy of ABC Enterprises?"
    search_results = vector_search(user_query, top_k=4)
    combined_prompt = f"You are an online store customer service agent who answers customer queries. Use context to answer the question. Only answer based on the provided context. \n\nUser query: {user_query}\n\nContext:\n1. {search_results[0]}\n2. {search_results[1]}\n3. {search_results[2]}\n4. {search_results[3]}\n\nAnswer the query based on the information provided."
    response = chat_with_model(combined_prompt)
    print(combined_prompt)
    print("Model Response:", response)
