import fitz
import pytesseract
from PIL import Image
import io
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama


# ==========================
# Chunking Function
# ==========================
def chunk_text(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================
# PDF Path
# ==========================
pdf_path = "documents/who.pdf"


# ==========================
# Open PDF
# ==========================
doc = fitz.open(pdf_path)


# ==========================
# Load Embedding Model
# ==========================
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ==========================
# Extract Text + OCR
# ==========================
print("\nExtracting PDF content...")

all_chunks = []

# Start from page 3
for page_num in range(2, len(doc)):

    print(f"Processing page {page_num + 1}")

    page = doc[page_num]

    # Normal PDF text
    page_text = page.get_text("text")

    # OCR text
    image_list = page.get_images(full=True)

    ocr_text = ""

    for img in image_list:

        try:
            xref = img[0]

            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            extracted_text = pytesseract.image_to_string(
                image
            ).strip()

            ocr_text += extracted_text + "\n"

        except Exception as e:
            print("OCR Error:", e)

    # Combine text + OCR
    combined_text = page_text + "\n" + ocr_text

    # Chunking
    chunks = chunk_text(
        combined_text,
        chunk_size=500,
        overlap=100
    )

    all_chunks.extend(chunks)


print("\nPDF extraction completed")
print(f"Total chunks created: {len(all_chunks)}")


# ==========================
# Create Embeddings
# ==========================
print("\nCreating embeddings...")

embeddings = embedding_model.encode(
    all_chunks,
    convert_to_numpy=True
)

embeddings = np.array(
    embeddings,
    dtype="float32"
)

print("Embeddings created")
print("Embedding shape:", embeddings.shape)


# ==========================
# Create FAISS Index
# ==========================
print("\nCreating FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index created")
print("Total vectors:", index.ntotal)


# ==========================
# Ask Questions
# ==========================
print("\nRAG system ready!")
print("Type 'exit' to quit.")

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    # Create query embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    # Search relevant chunks from FAISS
    distances, indices = index.search(
        query_embedding,
        k=5
    )

    # Build context
    context = ""

    for idx in indices[0]:
        context += all_chunks[idx] + "\n\n"

    # Prompt for Ollama
    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not found in the context,
say: "I could not find relevant information."

Context:
{context}

Question:
{query}

Answer:
"""

    print("\nThinking...\n")

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(response["message"]["content"])

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    # Create query embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    # Search in FAISS
    distances, indices = index.search(
        query_embedding,
        k=5
    )

    print("\nTop Relevant Chunks:\n")

    for i, idx in enumerate(indices[0]):

        print("=" * 80)
        print(f"Result {i + 1}")
        print("=" * 80)

        print(all_chunks[idx][:700])
        print("\n")