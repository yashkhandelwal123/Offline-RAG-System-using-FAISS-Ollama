import fitz
import pytesseract
from PIL import Image
import io
import faiss
import numpy as np
import pickle
import os

from sentence_transformers import SentenceTransformer


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
# Load Embedding Model
# ==========================
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ==========================
# Read PDFs
# ==========================
documents_folder = "documents"

all_chunks = []

pdf_files = [
    f for f in os.listdir(documents_folder)
    if f.endswith(".pdf")
]

print(f"Found {len(pdf_files)} PDFs")


for pdf_file in pdf_files:

    pdf_path = os.path.join(
        documents_folder,
        pdf_file
    )

    print(f"\nProcessing: {pdf_file}")

    doc = fitz.open(pdf_path)

    # Start from page 3
    for page_num in range(2, len(doc)):

        page = doc[page_num]

        page_text = page.get_text("text")

        image_list = page.get_images(
            full=True
        )

        ocr_text = ""

        for img in image_list:

            try:
                xref = img[0]

                base_image = doc.extract_image(
                    xref
                )

                image_bytes = base_image[
                    "image"
                ]

                image = Image.open(
                    io.BytesIO(
                        image_bytes
                    )
                )

                extracted_text = (
                    pytesseract.image_to_string(
                        image
                    ).strip()
                )

                ocr_text += (
                    extracted_text + "\n"
                )

            except Exception as e:
                print(
                    "OCR Error:",
                    e
                )

        combined_text = (
            page_text +
            "\n" +
            ocr_text
        )

        chunks = chunk_text(
            combined_text,
            chunk_size=500,
            overlap=100
        )

        all_chunks.extend(chunks)


print(
    f"\nTotal chunks: {len(all_chunks)}"
)


# ==========================
# Create Embeddings
# ==========================
print(
    "\nCreating embeddings..."
)

embeddings = embedding_model.encode(
    all_chunks,
    convert_to_numpy=True
)

embeddings = np.array(
    embeddings,
    dtype="float32"
)

print(
    "Embeddings created"
)


# ==========================
# Create FAISS Index
# ==========================
print(
    "\nCreating FAISS index..."
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)

print(
    "FAISS index created"
)


# ==========================
# Save Index
# ==========================
faiss.write_index(
    index,
    "faiss_index.bin"
)

print(
    "FAISS index saved"
)


# ==========================
# Save Chunks
# ==========================
with open(
    "chunks.pkl",
    "wb"
) as f:

    pickle.dump(
        all_chunks,
        f
    )

print("Chunks saved")

print(
    "\nBuild completed!"
)