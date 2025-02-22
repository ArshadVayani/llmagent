import faiss
import numpy as np
from openai import OpenAI

# Initialize OpenAI Client (Adjust base_url for local Ollama if needed)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def get_embedding(text):
    response = client.embeddings.create(
        model="all-minilm",
        input=text
    )
    return response.data[0].embedding

# Load FAISS index
index = faiss.read_index("policy_index.faiss")

# Load policy text
with open("policy_texts.txt", "r") as f:
    policy_texts = f.readlines()

def search_policy(query):
    """Search FAISS for the most relevant policy texts."""
    query_embedding = np.array([get_embedding(query)]).astype('float32')
    _, index_result = index.search(query_embedding, 4)  # Retrieve top 4 matches
    return [policy_texts[index_result[0][i]] for i in range(4)]
    #return policy_texts[index_result[0][0]]

# Example query
query = "Are there any shipping discounts?"
best_matches = search_policy(query)
print("Best matching policy sections:", best_matches)
