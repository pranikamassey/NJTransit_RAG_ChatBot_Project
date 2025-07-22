import json
import faiss
import numpy as np
import openai
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.common import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

# load FAISS + metadata
INDEX_PATH = "data/index.faiss"
META_PATH  = "data/meta.json"
index    = faiss.read_index(INDEX_PATH)
with open(META_PATH) as f:
    metadata = json.load(f)

app = FastAPI()
class Query(BaseModel):
    question: str

# pre‑compile regexes
GREETING_RE = re.compile(r"\b(hi+|hello|hey)\b", re.I)
DONT_KNOW_RE = re.compile(
    r"\b(i\s*(?:don'?t|do\s+not)\s+know|i\s+am\s+not\s+sure)\b",
    re.I
)

@app.post("/query")
async def query(body: Query):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Empty question")

    # 0) Quick greeting handler
    if GREETING_RE.search(q):
        return {"answer": "Hi there! I’m the Access Link bot—how can I help you today?"}

    # 1) Embed the question
    emb = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=q
    )
    q_emb = emb.data[0].embedding

    # 2) Retrieve top‑5 chunks
    D, I = index.search(np.array([q_emb], dtype="float32"), k=5)
    excerpts = [
        f"(p.{metadata[idx]['page']}) {metadata[idx]['text']}"
        for idx in I[0]
    ]

    # 3) Build prompt
    prompt  = (
        "You are an expert on NJ Transit Access Link policies.\n"
        "Use ONLY the following excerpts to answer. "
        "If the answer is not contained, say 'I don't know.'\n\n"
    )
    for i, ex in enumerate(excerpts, start=1):
        prompt += f"Excerpt {i}: {ex}\n\n"
    prompt += f"Question: {q}\nAnswer:"

    # 4) Call the LLM
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=300
    )
    answer = resp.choices[0].message.content.strip()

    # 5) Fallback for any “I don’t know” style reply
    if DONT_KNOW_RE.search(answer):
        answer = (
            "I’m sorry—I couldn’t find that information in the Access Link guidelines. "
            "For the most accurate answer, please call Access Link Customer Service at "
            "(973) 491‑4224 (option 5) or email adaservices@njtransit.com."
        )

    return {"answer": answer}
