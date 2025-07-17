# src/api.py
import json
import faiss
import numpy as np
import openai

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.common import OPENAI_API_KEY

# 1) Configure OpenAI
openai.api_key = OPENAI_API_KEY

# 2) Load FAISS + metadata
INDEX_PATH = "data/index.faiss"
META_PATH  = "data/meta.json"

index = faiss.read_index(INDEX_PATH)
with open(META_PATH) as f:
    metadata = json.load(f)

# 3) FastAPI setup
app = FastAPI()
class Query(BaseModel):
    question: str

@app.post("/query")
async def query(body: Query):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Empty question")

    # 4) Embed user query
  
    
    emb_resp = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=q
    )
    # extract the embedding vector
    q_emb = emb_resp.data[0].embedding


    # 5) Search FAISS
    D, I = index.search(np.array([q_emb], dtype="float32"), k=5)
    hits = I[0]

    # 6) Gather excerpts
    excerpts = [
        f"(p.{metadata[idx]['page']}) {metadata[idx]['text']}"
        for idx in hits
    ]

    # 7) Build prompt
    prompt = (
        "You are an expert on NJ Transit Access Link policies.\n"
        "Use ONLY the following excerpts to answer. If the answer is not contained, say 'I don't know.'\n\n"
    )
    for i, ex in enumerate(excerpts, start=1):
        prompt += f"Excerpt {i}: {ex}\n\n"
    prompt += f"Question: {q}\nAnswer:"

    # 8) Call ChatCompletion

    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300
    )
    answer = resp.choices[0].message.content.strip()

    return {"answer": answer}
