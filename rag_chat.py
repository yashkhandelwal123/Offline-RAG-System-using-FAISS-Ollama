import faiss
import pickle
import numpy as np
from ollama import Client

from sentence_transformers import (
    SentenceTransformer
)


# ==========================
# Load Model
# ==========================
print(
    "Loading embedding model..."
)

embedding_model = (
    SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
)

print(
    "Embedding model loaded"
)


# ==========================
# Load FAISS Index
# ==========================
print(
    "Loading FAISS index..."
)

index = faiss.read_index(
    "faiss_index.bin"
)

print(
    "FAISS loaded"
)


# ==========================
# Load Chunks
# ==========================
print(
    "Loading chunks..."
)

with open(
    "chunks.pkl",
    "rb"
) as f:

    all_chunks = pickle.load(
        f
    )

print(
    "Chunks loaded"
)

client = Client(
    host="http://192.168.1.10:11434"
)


print("\nRAG Ready!")
print("Type exit to quit.")


# ==========================
# Chat Loop
# ==========================
while True:

    query = input(
        "\nAsk question: "
    )

    if query.lower() == "exit":
        break

    query_embedding = (
        embedding_model.encode(
            [query],
            convert_to_numpy=True
        )
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    distances, indices = (
        index.search(
            query_embedding,
            k=5
        )
    )

    context = ""

    for idx in indices[0]:

        context += (
            all_chunks[idx]
            + "\n\n"
        )

    prompt = f"""
You are a helpful assistant.

Answer only using
the provided context.

If the answer is
not in context,
say:
"I could not find
relevant information."

Context:
{context}

Question:
{query}

Answer:
"""

    print(
        "\nThinking..."
    )

    response = client.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\n")
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(
        response[
            "message"
        ]["content"]
    )