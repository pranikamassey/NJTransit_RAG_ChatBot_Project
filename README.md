NJT Access Link RAG Chatbot - Project Documentation
Project developed by - Pranika Massey (Summer Intern at NJ Transit)
A complete, end‑to‑end guide for setup, build, deployment, usage, and maintenance.

1) What this project is
An information assistant for NJ TRANSIT Access Link riders and staff. It uses Retrieval‑Augmented Generation (RAG) to answer questions from official PDFs and pages (e.g., Access Link Customer Guidelines, brochures, accessibility pages). Users ask questions in a chat UI; the backend retrieves relevant text chunks from a local FAISS index and the LLM answers only from those excerpts (with a safety fallback to “Contact support” when needed).

2) Other Uses:
Immediate answers for common questions (eligibility, booking, fares, cancellations, no‑shows, etc.).
Consistency: Answers are grounded in official documents (reduced risk of policy drift).
Reduced call volume: Handles FAQs before they reach human agents.
Extendable: Add more NJT sub‑domains or documents as the service grows.

3) Prerequisites
Python 3.10+ recommended
macOS, Linux, or Windows (PowerShell)
An OpenAI API key with access to embeddings + chat models
(Optional) UVicorn for local API serving (installed from requirements.txt)

4) STEPS TO RUN THE CODE
    # 1) clone + enter
    git clone <your-repo-url>
    cd NJTRANSIT_RAG_CHATBOT

    # 2) create & activate venv
    python -m venv .venv
    # macOS/Linux:
    source .venv/bin/activate
    # Windows (PowerShell):
    .venv\Scripts\Activate.ps1

    # 3) install
    pip install -r requirements.txt

    # 4) configure secrets
    cp .env.example .env
    # then edit .env and set OPENAI_API_KEY=<your-key>

    # 5) ingest documents (build FAISS index)
    python -m src.ingest --pdf data/ \
    --index-output data/index.faiss \
    --meta-output data/meta.json

    # 6) start the API
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

    # 7) start the UI (new terminal)
    python app.py
    # open the shown URL; default http://127.0.0.1:7861

5) Configuration

    Environment variables (.env)
    OPENAI_API_KEY=sk-...your key...

    Gradio UI settings (optional)
    GRADIO_SERVER_PORT (defaults to 7861)

    API URL (UI → Backend)
    app.py uses API_URL = "http://127.0.0.1:8000/query".
    Change this if the API runs elsewhere (e.g., Docker, cloud, different port).

6) How it works (architecture + data flow)
    1) Ingestion (src/ingest.py)
        Inputs: PDFs in data/ and (optionally) public web URLs.
        Parsing:
            PDFs via pdfplumber page‑by‑page.
            Web pages via requests + BeautifulSoup (scripts/styles/headers/footers removed).
        Chunking: Word‑based sliding windows (max_tokens≈400, overlap ≈50 words).
        Embedding: Each chunk is embedded with OpenAI embeddings.
        Indexing: Chunks are stored in a FAISS index (index.faiss). Sidecar meta.json keeps {source, page, text} for each vector.

    2) Retrieval API (src/api.py)
        /query (POST): Accepts { "question": "..." }.
        Embeds the question, searches FAISS for top‑K chunks (K=5).
        Builds a prompt with those excerpts and calls the chat model.
        Returns { "answer": "..." }. If the answer indicates uncertainty, returns a contact fallback message.

    3) Frontend (app.py)
        Gradio chat UI that stores message history on the client.
        Sends the user’s latest message to /query, displays the answer.
        Includes a greeting color scheme and “Clear Chat” control.

7) Ingestion Pipeline(in detail)
    # PDFs from a folder
        python -m src.ingest --pdf data/ \
        --index-output data/index.faiss \
        --meta-output data/meta.json

    # PDFs + multiple URLs
        python -m src.ingest --pdf data/ \
        --url https://www.njtransit.com/accessibility \
            https://www.njtransit.com/accessibility/access-link-ada-paratransit \
        --index-output data/index.faiss \
        --meta-output data/meta.json

    Adding/removing sources
        Drop new PDFs into data/ (or pass explicit --pdf path/to/file.pdf).
        Add URLs through --url ....
        Re-run the ingest command to rebuild index.faiss and meta.json.

    meta.json format
        Each embedding has a parallel metadata entry:
        {
            "source": "AccessLinkCustomerGuidelines.pdf",
            "page": 12,
            "text": "Chunk text here..."
        }
        Note: For URLs, page is an incrementing counter per page-equivalent chunk.

8) API (backend) specification Endpoint

    Endpoint
    POST /query

    Request:
    {
        "question": "What is the no-show policy?"
    }
    Response(200):
    {
        "answer": "Your grounded answer here..."
    }

    Running the API locally
        uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

9) Using the frontend

    Start with python app.py.
    Type a question (e.g., “How do I apply for Access Link?”).
    You’ll see user and assistant bubbles with custom colors.
    Click Clear Chat to reset the conversation.

10) How to use (typical flows)
    Ask direct questions:
    “How do I schedule a ride?”, “What are the eligibility criteria?”, “What counts as a no‑show?”

    The bot:
    Detects greetings (“Hi”, “Hello”…) and responds politely.
    Retrieves top 5 relevant excerpts.
    Answers using only the provided excerpts.
    If unsupported/unclear, provides a contact fallback (phone/email).

11) Optional Improvements

    These are optional improvements you can adopt over time:

    - Similarity choice
    Current index uses L2 distance on raw embeddings. For better ranking stability:

        L2‑normalize embeddings and queries and switch to faiss.IndexFlatIP (cosine similarity), 
        or normalize while keeping IndexFlatL2.

    - Citations in answers
    Return the top source(s) with page numbers to the UI (e.g., sources: [{source, page}]) so users can verify.

    - “I don’t know” threshold
    Add a min‑similarity (or max‑distance) threshold before answering. If below threshold → return the contact fallback.

    - Batch embeddings
    In ingest.py, send input=[chunk1, chunk2, ...] to embed 64–256 chunks per call to speed up indexing.

    - Multi‑turn context
    Include a short chat history window or a “question rewriter” step so follow‑ups are contextual.

    - Prompt hardening
    Use a system message with policies and anti‑prompt‑injection guidance.

    - Monitoring
    Log queries (redact PII), top‑K scores, and fallback rates. Track common unanswered questions to improve sources.




