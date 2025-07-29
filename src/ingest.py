# src/ingest.py


##steps for referencee
#source .venv/bin/activate



# python -m src.ingest \
#   --pdf data/ \
#   --url \
#     https://njtransit.com/accessibility \
#     https://www.njtransit.com/accessibility/access-link-ada-paratransit \
#     https://www.njtransit.com/accessibility/community-transportation \
#     https://www.njtransit.com/schedules-and-fares/reduced-fare-program \
#     https://www.njtransit.com/magnusmode \
#     https://www.njtransit.com/accessibility/voter-registration-application \
#     https://www.njtransit.com/ffy-2024-grant-application \
#   --index-output data/index.faiss \
#   --meta-output data/meta.json



import argparse
import json
import os

import faiss
import numpy as np
import pdfplumber
import openai
import requests
from bs4 import BeautifulSoup


from src.common import OPENAI_API_KEY

def chunk_text(text, max_tokens=400, overlap=50):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max_tokens - overlap
    return chunks

def main(pdf_inputs, url_inputs, index_output, meta_output):
    # pdf_inputs: list of file-or-dir paths
    print(f"🟢 Ingesting {len(pdf_inputs)} PDF(s) + {len(url_inputs)} URL(s)…")
    

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
    # --- Web ingestion, new! ---
    for link in url_inputs:
        print(f"🌐 Fetching {link}")
        resp = requests.get(link, timeout=10)
        resp.raise_for_status()
        # parse visible text
        soup = BeautifulSoup(resp.text, "lxml")
        # remove scripts/styles
        for tag in soup(["script","style","header","footer","nav","aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # optional: split on paragraphs
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        combined = "\n\n".join(paras)
        # chunk & record metadata
        for i, chunk in enumerate(chunk_text(combined), start=1):
            all_chunks.append(chunk)
            meta.append({
                "source": link,
                "page": i,           # or None, or 0 — but key must exist
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
        nargs="*",
        default=[],
        help="PDF files or directories to ingest"
    )
    parser.add_argument(
        "--url",
        required=False,
        nargs="+",
        default=[],
        help="One or more public webpage URLs to ingest"
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
    main(
        pdf_inputs   = args.pdf,
        url_inputs   = args.url,
        index_output = args.index_output,
        meta_output  = args.meta_output
    )
