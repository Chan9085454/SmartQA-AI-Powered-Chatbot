import os
import sys
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

def load_json_or_exit(path):
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    return json.load(open(path, "r", encoding="utf-8"))

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chunks_path = os.path.join(base_dir, "chunks.json")
    emb_path    = os.path.join(base_dir, "chunk_embeddings.json")

    # 1) load raw JSON
    chunks      = load_json_or_exit(chunks_path)
    emb_records = load_json_or_exit(emb_path)

    # 2) build a single NumPy array of embeddings as float32
    np_embs = np.array([rec["embedding"] for rec in emb_records], dtype=np.float32)

    # 3) convert to a PyTorch tensor
    chunk_embeddings = torch.from_numpy(np_embs)  # dtype=torch.float32

    # 4) load the SBERT model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[OK] Loaded model and data. Type your question or 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in ("exit","quit"):
            print("Goodbye!")
            break

        # 5) encode query directly as a PyTorch tensor (float32)
        q_emb = model.encode(query, convert_to_tensor=True)  # dtype=torch.float32

        # 6) compute cosine similarities
        scores = util.cos_sim(q_emb, chunk_embeddings)[0]

        # 7) grab top 3
        top_idxs = scores.argsort(descending=True)[:3]

        print("\nTop matching chunks:\n")
        for rank, idx in enumerate(top_idxs, start=1):
            score = scores[idx].item()
            text  = chunks[idx]["text"].replace("\n"," ")
            snippet = text[:200] + ("…" if len(text) > 200 else "")
            print(f"{rank}. [Chunk {chunks[idx]['id']}] (score={score:.4f})")
            print(f"   {snippet}\n")
        print("-" * 40)

if __name__ == "__main__":
    main()
