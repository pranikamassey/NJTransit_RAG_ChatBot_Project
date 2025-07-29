# src/ingest.py


##steps for referencee
#source .venv/bin/activate

## Ingest the first PDF, overwriting old index/meta:
# python src/ingest.py \
#   --pdf data/AccessLinkCustomerGuidelines.pdf \
#   --index-output data/index.faiss \
#   --meta-output data/meta.json

## Ingest the second PDF, but this time *append* to the existing index/meta:
# python src/ingest.py \
#   --pdf data/AnotherPolicyDoc.pdf \
#   --index-output data/index.faiss \
#   --meta-output data/meta.json \
#   --append



import argparse
import json
import os

import faiss
import numpy as np
import pdfplumber
import openai

from src.common import OPENAI_API_KEY

def chunk_text(text, max_tokens=400, overlap=50):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max_tokens - overlap
    return chunks

def main(pdf_inputs, index_output, meta_output):
    # pdf_inputs: list of file-or-dir paths
    print(f"🟢 Starting ingestion of {len(pdf_inputs)} input path(s)…")

    # 1) Collect all PDF files
    pdf_files = []
    for p in pdf_inputs:
        if os.path.isdir(p):
            for fn in os.listdir(p):
                if fn.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(p, fn))
        elif p.lower().endswith(".pdf") and os.path.isfile(p):
            pdf_files.append(p)
        else:
            print(f"⚠️  Skipping non-PDF path: {p}")

    if not pdf_files:
        print("❌ No PDFs found to ingest. Exiting.")
        return

    # 2) Read & chunk each PDF
    all_chunks = []
    meta       = []
    for pdf_file in pdf_files:
        print(f"📄 Ingesting {pdf_file}")
        with pdfplumber.open(pdf_file) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                for chunk in chunk_text(text):
                    all_chunks.append(chunk)
                    meta.append({
                        "source": os.path.basename(pdf_file),
                        "page": page_no,
                        "text": chunk
                    })
    print(f"🔸 Extracted {len(all_chunks)} total chunks.")

    # 2) Embed via new API
    openai.api_key = OPENAI_API_KEY
    embeddings = []
    for i, chunk in enumerate(all_chunks, start=1):
        
        resp = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=chunk
        )
    # pull out the embedding from the response object
        embedding = resp.data[0].embedding
        embeddings.append(embedding)

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
    parser = argparse.ArgumentParser(
        description="Ingest one or more PDFs (or directories of PDFs) into a single FAISS index."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        nargs="+",
        help="One or more paths: either PDF files or folders containing PDFs"
    )
    parser.add_argument(
        "--index-output",
        default="data/index.faiss",
        help="Where to write the combined FAISS index"
    )
    parser.add_argument(
        "--meta-output",
        default="data/meta.json",
        help="Where to write the combined metadata JSON"
    )
    args = parser.parse_args()
    main(args.pdf, args.index_output, args.meta_output)
