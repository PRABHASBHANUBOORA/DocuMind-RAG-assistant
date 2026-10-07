"""
DocuMind — Multi-PDF RAG Q&A Assistant
----------------------------------------
Upload one or more PDFs, ask questions, and get streamed answers grounded
in the document content, with page citations, source excerpts, and
conversation-aware follow-up suggestions.

Stack:
- Streamlit        -> UI
- PyPDF             -> PDF text extraction
- LangChain         -> chunking
- HuggingFace       -> local, free sentence-embeddings (no API key needed)
- NumPy             -> per-document cosine-similarity ranking (no vector DB)
- Groq              -> free, fast LLM inference for answers + suggestions
"""

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import re
import requests
import numpy as np
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document

st.set_page_config(page_title="DocuMind — RAG PDF Assistant", page_icon="📄", layout="wide")

# ---------- Custom styling ----------
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top left, #1E293B 0%, #0F172A 55%);
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #94A3B8;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.2rem;
    }
    .stat-badge {
        display: inline-block;
        background-color: #1E293B;
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 6px 14px;
        margin: 2px 6px 10px 0;
        font-size: 0.82rem;
    }
    .stat-badge b {
        color: #38BDF8;
    }
    .source-pill {
        display: inline-block;
        background-color: #172554;
        color: #CBD5E1;
        border-radius: 999px;
        padding: 2px 12px;
        margin: 2px 4px 2px 0;
        font-size: 0.78rem;
        border: 1px solid #334155;
    }
    div[data-testid="stChatMessage"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.6rem;
    }
    .empty-state {
        text-align: center;
        padding: 3.5rem 1rem;
        color: #94A3B8;
    }
    .empty-state .big-emoji {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .followup-label {
        color: #8B5CF6;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.6rem 0 0.3rem 0;
    }
    .excerpt-block {
        background-color: #172554;
        border-left: 3px solid #8B5CF6;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #CBD5E1;
    }
    .excerpt-source {
        color: #38BDF8;
        font-weight: 600;
        font-size: 0.78rem;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] > div:first-child,
    div[data-testid="stSidebarContent"] {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }
    .sidebar-footer {
        margin-top: auto;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        color: #94A3B8;
        font-size: 0.78rem;
        border-top: 1px solid #334155;
    }
    div.stButton > button {
        border-radius: 999px;
        border: 1px solid #334155;
        background-color: #1E293B;
        color: #F8FAFC;
    }
    div.stButton > button:hover {
        border-color: #38BDF8;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📄 DocuMind</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about your PDFs — answers are grounded in the document, with page citations.</p>', unsafe_allow_html=True)

groq_api_key = st.secrets.get("GROQ_API_KEY", None)
supabase_url = st.secrets.get("SUPABASE_URL", None)
supabase_key = st.secrets.get("SUPABASE_ANON_KEY", None)

if supabase_url:
    # Normalize: strip trailing slashes and any accidentally-included
    # /rest/v1 suffix, since we append that path ourselves below.
    supabase_url = supabase_url.rstrip("/")
    for suffix in ("/rest/v1", "/rest"):
        if supabase_url.endswith(suffix):
            supabase_url = supabase_url[: -len(suffix)]

# ---------- Session state ----------
for key, default in [("doc_chunks", None), ("embeddings", None), ("messages", []), ("doc_names", []),
                      ("followups", []), ("feedback", {}), ("stats", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📁 Your documents")

    if not groq_api_key:
        st.warning("No Groq key configured by the app owner. Paste your own to use this app.")
        groq_api_key = st.text_input("Groq API key", type="password")
    else:
        st.success("Ready to go — you can upload your documents in the below upload session in Upload PDF(s) to get the info about the doucment.")

    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    build_button = st.button("✨ Process documents", use_container_width=True)

    if st.session_state.messages:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.followups = []
            st.session_state.feedback = {}
            st.rerun()

    st.markdown(
        '<div class="sidebar-footer">Built with LangChain, HuggingFace embeddings, and Groq.</div>',
        unsafe_allow_html=True,
    )


def extract_documents(files):
    docs = []
    for f in files:
        reader = PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": f.name, "page": page_num}))
    return docs


def get_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def retrieve(question):
    """Rank each uploaded document's own chunks by similarity to the
    question, independently per document, then combine the results.

    This replaced an earlier approach using FAISS's built-in metadata
    `filter` on a single shared index, which intermittently returned zero
    results for a specific document on some calls and worked fine on
    others — a silent, hard-to-diagnose failure. Doing the similarity
    ranking ourselves with plain cosine similarity is slightly more code,
    but fully deterministic and transparent: every document's chunks are
    guaranteed to be considered every single time, with no hidden
    filtering step that can fail quietly.
    """
    doc_chunks = st.session_state.doc_chunks
    embeddings = st.session_state.embeddings
    if not doc_chunks or embeddings is None:
        return []

    question_vector = embeddings.embed_query(question)
    doc_names = st.session_state.doc_names or list(doc_chunks.keys())
    per_doc_k = max(3, 12 // max(len(doc_names), 1))

    results = []
    for name in doc_names:
        pairs = doc_chunks.get(name, [])
        ranked = sorted(pairs, key=lambda cv: cosine_similarity(question_vector, cv[1]), reverse=True)
        results.extend(chunk for chunk, _ in ranked[:per_doc_k])
    return results


def generate_followups(question, answer):
    try:
        llm = get_llm()
        prompt = (
            "Based on this question-and-answer exchange about an uploaded document, "
            "suggest exactly 3 short, natural follow-up questions the user might ask "
            "next. Keep each under 10 words. Return ONLY a numbered list, nothing else.\n\n"
            f"Question: {question}\nAnswer: {answer}"
        )
        result = llm.invoke(prompt).content
        lines = [re.sub(r"^\d+[\.\)]\s*", "", l).strip() for l in result.split("\n") if l.strip()]
        return [l for l in lines if l][:3]
    except Exception:
        return []


def render_sources_and_excerpts(sources, relevant_docs, key_prefix):
    st.markdown(
        " ".join(f'<span class="source-pill">{s}</span>' for s in sources),
        unsafe_allow_html=True,
    )
    with st.expander("📚 View source excerpts"):
        seen = set()
        for doc in relevant_docs:
            label = f"{doc.metadata['source']} · p.{doc.metadata['page']}"
            if label in seen:
                continue
            seen.add(label)
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220].rsplit(" ", 1)[0] + "..."
            st.markdown(
                f'<div class="excerpt-block"><span class="excerpt-source">{label}</span><br>{snippet}</div>',
                unsafe_allow_html=True,
            )


def log_feedback(question, answer, rating):
    """Write one feedback row to Supabase. Returns True on success.
    Stores a diagnostic message in session_state on failure for debugging."""
    if not supabase_url or not supabase_key:
        st.session_state["last_feedback_error"] = "No SUPABASE_URL / SUPABASE_ANON_KEY found in secrets."
        return False
    try:
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        payload = {
            "question": question,
            "answer": answer[:2000],
            "rating": rating,
            "documents": ", ".join(st.session_state.doc_names),
        }
        resp = requests.post(
            f"{supabase_url}/rest/v1/feedback", json=payload, headers=headers, timeout=5
        )
        if resp.status_code in (200, 201):
            return True
        st.session_state["last_feedback_error"] = f"HTTP {resp.status_code}: {resp.text[:300]}"
        return False
    except Exception as e:
        st.session_state["last_feedback_error"] = f"{type(e).__name__}: {e}"
        return False


def render_feedback(msg_index, question, answer):
    fb = st.session_state.feedback.get(msg_index)
    c1, c2, c3 = st.columns([1, 1, 10])
    if c1.button("👍", key=f"up_{msg_index}"):
        saved = log_feedback(question, answer, "up")
        st.session_state.feedback[msg_index] = "up" if saved else "up_unsaved"
        st.rerun()
    if c2.button("👎", key=f"down_{msg_index}"):
        saved = log_feedback(question, answer, "down")
        st.session_state.feedback[msg_index] = "down" if saved else "down_unsaved"
        st.rerun()
    if fb == "up":
        c3.caption("Thanks for the feedback! 🙌 Saved.")
    elif fb == "down":
        c3.caption("Thanks — noted for improvement. Saved.")
    elif fb in ("up_unsaved", "down_unsaved"):
        c3.caption("Feedback recorded for this session only.")
        err = st.session_state.get("last_feedback_error")
        if err:
            c3.caption(f"⚠️ Debug: {err}")


def answer_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            relevant_docs = retrieve(question)
            context = "\n\n".join(
                f"[Source: {doc.metadata['source']}, page {doc.metadata['page']}]\n{doc.page_content}"
                for doc in relevant_docs
            )
            doc_list = ", ".join(st.session_state.doc_names)
            prompt = (
                f"The user has uploaded these documents: {doc_list}. "
                "If the question asks which documents exist, whether a particular "
                "file was uploaded, or similar meta-questions about the file list "
                "itself, answer directly from that list above.\n\n"
                "For questions about the CONTENT of the documents, answer using "
                "only the context below. Each chunk of context is labeled with "
                "the exact source document it came from. Never attribute a fact "
                "to a document other than the one it is labeled with, and never "
                "mix or combine details from different source documents when "
                "describing a single person or item. If the answer isn't in the "
                "context, say you don't know.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            )
            llm = get_llm()

        # Stream the answer token-by-token for a livelier feel
        def stream_chunks():
            full = ""
            for chunk in llm.stream(prompt):
                full += chunk.content
                yield chunk.content
            stream_chunks.full_answer = full

        answer_text = st.write_stream(stream_chunks)

        seen, sources = set(), []
        for doc in relevant_docs:
            label = f"{doc.metadata['source']} · p.{doc.metadata['page']}"
            if label not in seen:
                seen.add(label)
                sources.append(label)
        render_sources_and_excerpts(sources, relevant_docs, key_prefix=len(st.session_state.messages))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources,
        "relevant_docs": [(d.metadata["source"], d.metadata["page"], d.page_content) for d in relevant_docs],
    })
    render_feedback(len(st.session_state.messages) - 1, question, answer_text)
    st.session_state.followups = generate_followups(question, answer_text)


# ---------- Process documents ----------
if build_button:
    if not uploaded_files:
        st.sidebar.warning("Upload at least one PDF first.")
    else:
        with st.status("Processing your documents...", expanded=True) as status:
            st.write("📖 Reading PDF pages...")
            raw_docs = extract_documents(uploaded_files)
            st.write("✂️ Splitting into chunks...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(raw_docs)
            st.write("🧠 Creating embeddings (first run downloads a small model)...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.write("🗂️ Indexing chunks per document...")
            chunk_vectors = embeddings.embed_documents([c.page_content for c in chunks])
            doc_chunks = {}
            for chunk, vector in zip(chunks, chunk_vectors):
                doc_chunks.setdefault(chunk.metadata["source"], []).append((chunk, vector))
            st.session_state.doc_chunks = doc_chunks
            st.session_state.embeddings = embeddings
            st.session_state.messages = []
            st.session_state.followups = []
            st.session_state.feedback = {}
            st.session_state.doc_names = [f.name for f in uploaded_files]
            st.session_state.stats = {
                "documents": len(uploaded_files),
                "pages": len(raw_docs),
                "chunks": len(chunks),
            }
            status.update(label=f"Indexed {len(uploaded_files)} document(s) ✅", state="complete", expanded=False)

# ---------- Main area ----------
if st.session_state.doc_chunks is None:
    st.markdown("""
    <div class="empty-state">
        <div class="big-emoji">🗂️</div>
        <b>No documents indexed yet</b><br>
        Upload one or more PDFs in the sidebar and click <b>Process documents</b> to get started.
    </div>
    """, unsafe_allow_html=True)
else:
    if st.session_state.stats:
        s = st.session_state.stats
        st.markdown(
            f'<span class="stat-badge">📄 Documents: <b>{s["documents"]}</b></span>'
            f'<span class="stat-badge">📃 Pages: <b>{s["pages"]}</b></span>'
            f'<span class="stat-badge">🧩 Chunks: <b>{s["chunks"]}</b></span>',
            unsafe_allow_html=True,
        )
    if st.session_state.doc_names:
        st.markdown(
            " ".join(f'<span class="source-pill">📄 {n}</span>' for n in st.session_state.doc_names),
            unsafe_allow_html=True,
        )

    # Chat history (already-answered turns)
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                docs_like = [
                    type("D", (), {"metadata": {"source": s, "page": p}, "page_content": c})()
                    for (s, p, c) in msg.get("relevant_docs", [])
                ]
                render_sources_and_excerpts(msg["sources"], docs_like, key_prefix=i)
                prior_question = st.session_state.messages[i - 1]["content"] if i > 0 else ""
                render_feedback(i, prior_question, msg["content"])

    # Starter suggestions (before any question is asked)
    if not st.session_state.messages:
        st.caption("Try asking:")
        starters = [
            "Summarize this document in 3 bullet points",
            "What are the key skills mentioned?",
            "List every name mentioned across the documents",
        ]
        cols = st.columns(len(starters))
        for col, s in zip(cols, starters):
            if col.button(s, use_container_width=True, key=f"start_{s}"):
                if not groq_api_key:
                    st.warning("A Groq API key is needed to answer questions.")
                else:
                    answer_question(s)
                    st.rerun()

    # Dynamic, conversation-aware follow-up suggestions (Copilot-style)
    elif st.session_state.followups:
        st.markdown('<p class="followup-label">💡 You might also ask:</p>', unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.followups))
        for col, s in zip(cols, st.session_state.followups):
            if col.button(s, use_container_width=True, key=f"followup_{s}"):
                answer_question(s)
                st.rerun()

# ---------- Chat input ----------
question = st.chat_input("Ask a question about your documents...")

if question:
    if st.session_state.doc_chunks is None:
        st.warning("Upload and process at least one PDF first.")
    elif not groq_api_key:
        st.warning("A Groq API key is needed to answer questions.")
    else:
        answer_question(question)
        st.rerun()
