from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key='ollama'
)

response = client.embeddings.create(
    model='all-minilm',
    input='Hello, world!'
)

print(response.data[0].embedding)