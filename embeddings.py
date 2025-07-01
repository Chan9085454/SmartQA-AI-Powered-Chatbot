import json
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load chunks from chunks.json
chunks_path = "chunks.json"
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Extract texts for embedding
texts = [chunk["text"] for chunk in chunks]

# Load pre-trained SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings with progress bar
print("[INFO] Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Prepare embedding data
embedding_data = [
    {"id": chunk["id"], "embedding": embedding.tolist()}
    for chunk, embedding in zip(chunks, embeddings)
]

# Save embeddings to a JSON file
output_path = os.path.join("chunk_embeddings.json")
try:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embedding_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(embedding_data)} embeddings to {output_path}")
except Exception as e:
    print(f"[ERROR] Could not save embeddings: {e}")
