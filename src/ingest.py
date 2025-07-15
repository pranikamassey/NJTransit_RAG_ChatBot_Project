# src/ingest.py
import argparse
import json
import os

import faiss
import numpy as np
import pdfplumber
import openai

from common import OPENAI_API_KEY

def chunk_text(text, max_tokens=400, overlap=50):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max_tokens - overlap
    return chunks

def main(pdf_path, index_output, meta_output):
    print("🟢 Starting ingestion…")

    # 1) Read & chunk
    all_chunks, meta = [], []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for chunk in chunk_text(text):
                all_chunks.append(chunk)
                meta.append({"page": page_no, "text": chunk})
    print(f"🔸 Extracted {len(all_chunks)} chunks.")

    # 2) Embed via new API
    openai.api_key = OPENAI_API_KEY
    embeddings = []
    for i, chunk in enumerate(all_chunks, start=1):
        resp = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        embeddings.append(resp["data"][0]["embedding"])
        if i % 50 == 0:
            print(f"🔹 Embedded {i}/{len(all_chunks)}")
    print("🔵 Embeddings complete.")

    # 3) Build FAISS index
    dim   = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    print("🔶 FAISS index built.")

    # 4) Save index & metadata
    os.makedirs(os.path.dirname(index_output), exist_ok=True)
    faiss.write_index(index, index_output)
    with open(meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"🟣 Wrote index to {index_output} and meta to {meta_output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdf",          required=True)
    p.add_argument("--index-output", default="data/index.faiss")
    p.add_argument("--meta-output",  default="data/meta.json")
    args = p.parse_args()
    main(args.pdf, args.index_output, args.meta_output)
