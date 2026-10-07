# DocuMind — Multi-PDF RAG Q&A Assistant

A Retrieval-Augmented Generation (RAG) application that lets you upload PDF
documents and have a real conversation about their content. Answers are
grounded in the actual document text, streamed live, cited by page, and
followed up with smart, conversation-aware suggested questions.

## Features

- 📄 Upload and query multiple PDFs at once
- 💬 Streamed, chat-style answers (not a static block of text)
- 🔗 Page-level source citations, with an expandable view of the exact
  excerpt each citation came from
- 🧭 MMR (Maximal Marginal Relevance) retrieval, which deliberately pulls
  *diverse* chunks across all uploaded documents — not just whichever single
  document scores highest — so multi-document questions get fair coverage
- 🏷️ Source-labeled context, so the model never blends facts between two
  different uploaded documents when answering
- 💡 Dynamic, Copilot-style follow-up question suggestions generated after
  every answer, based on that specific conversation
- 👍👎 Thumbs up/down feedback, persisted to a real Supabase database
- 📊 Live document/page/chunk stats once a document set is processed
- 🎨 Custom slate/sky-blue/violet theme

## How it works (RAG pipeline)

1. **Ingest** — PDFs are parsed page-by-page with `pypdf`.
2. **Chunk** — Text is split into overlapping ~1000-character chunks with
   LangChain's `RecursiveCharacterTextSplitter`, preserving page metadata.
3. **Embed** — Each chunk is converted into a vector using a free, local
   HuggingFace sentence-transformer (`all-MiniLM-L6-v2`) — no API cost.
4. **Store & retrieve** — Each chunk's embedding is kept in memory, grouped
   by source document. At query time, each document's chunks are ranked
   independently by **cosine similarity** to the question, so every
   uploaded document is guaranteed to contribute to the answer — a large
   document can never crowd out a smaller one. (An earlier version used a
   shared FAISS index with MMR search; see `PROJECT_DEEP_DIVE.md` for why
   that was replaced after an intermittent retrieval bug.)
5. **Generate** — Retrieved chunks are labeled by source document and page,
   then passed to **Groq's `openai/gpt-oss-120b`** model via LangChain,
   with explicit instructions never to mix facts across documents.
6. **Cite** — The source PDF filename and page number are shown next to
   every answer, with an optional expandable panel showing the exact
   excerpt text used.
7. **Suggest** — A lightweight follow-up call asks the same model to
   propose 3 short next questions based on the conversation so far.
8. **Log feedback** — Thumbs up/down clicks are written to a Supabase
   (Postgres) table over its REST API for later review.

## Run locally

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (see `secrets.toml.example` for the
format) with your Groq key, and optionally your Supabase URL/key if you
want persistent feedback:

```toml
GROQ_API_KEY = "your-groq-key"
SUPABASE_URL = "https://your-project-ref.supabase.co"
SUPABASE_ANON_KEY = "your-anon-public-key"
```

Then run:

```bash
streamlit run app.py
```

## Run with Docker

```bash
docker build -t documind .
docker run -p 8501:8501 documind
```

## Deploy for free

Push this repo to GitHub, then deploy on
[Streamlit Community Cloud](https://streamlit.io/cloud). Add the same
secrets (`GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`) in the app's
**Settings → Secrets** panel so visitors don't need their own key.

## Tech stack

Python · LangChain · HuggingFace Embeddings · NumPy · Groq (`openai/gpt-oss-120b`)
· Streamlit · Supabase · Docker

See `PROJECT_DEEP_DIVE.md` for a full explanation of *why* each piece of
this stack was chosen, and a set of likely interview questions about it.

## Suggested resume project bullets

> **DocuMind — Multi-PDF RAG Q&A Assistant (Python, LangChain, Groq)**
> - Built a Retrieval-Augmented Generation application enabling grounded,
>   streamed natural-language Q&A over multiple PDF documents, using
>   LangChain for chunking/orchestration and HuggingFace embeddings.
> - Diagnosed an intermittent, silent retrieval bug in a FAISS-based vector
>   search filter and replaced it with a deterministic per-document cosine
>   similarity ranking, guaranteeing every uploaded document contributes to
>   every answer regardless of relative document size.
> - Designed source-labeled prompting to prevent cross-document hallucination
>   when answering questions spanning multiple uploaded files.
> - Implemented dynamic, conversation-aware follow-up question generation
>   and persisted user feedback (thumbs up/down) to a Supabase database via
>   its REST API.
> - Containerized the application with Docker and deployed it on Streamlit
>   Community Cloud with secrets-based credential management.
