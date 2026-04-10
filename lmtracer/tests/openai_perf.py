import os

from openai import OpenAI

BASE_URL = os.getenv("BASE_URL", "http://localhost/v1")

client = OpenAI(
    api_key="EMPTY",
    base_url=BASE_URL
)

resp = client.embeddings.create(
    model="Qwen/Qwen3-Embedding-0.6B",
    input="hello world"
)

print(resp.data[0].embedding)