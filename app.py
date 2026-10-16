"""Streamlit UI for the Agentic RAG System.  Run:  streamlit run app.py"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from rag import KnowledgeBase, Toolbox, build_agent, load_directory, load_document
from rag.agent import DEFAULT_MODEL, OfflineAgent

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "data" / "documents"
STORAGE_DIR = ROOT / "storage"

st.set_page_config(page_title="Agentic RAG System", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Building knowledge base (extract → clean → chunk → embed → store)…")
def get_kb() -> KnowledgeBase:
    if (STORAGE_DIR / "store.pkl").exists():
        try:
            return KnowledgeBase.load(STORAGE_DIR)
        except Exception:
            pass
    kb = KnowledgeBase()
    kb.add_documents(load_directory(DOCS_DIR))
    kb.save(STORAGE_DIR)
    return kb


kb = get_kb()
if "toolbox" not in st.session_state:
    st.session_state.toolbox = Toolbox(kb, DOCS_DIR)
toolbox: Toolbox = st.session_state.toolbox
st.session_state.setdefault("messages", [])

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Anthropic API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""),
                            help="Leave empty to run the offline rule-based agent.")
    model = st.text_input("Claude model", value=DEFAULT_MODEL)
    effort = st.select_slider("Reasoning effort", ["low", "medium", "high", "xhigh"], value="medium")
    show_trace = st.toggle("Show agent trace", value=True)

    st.divider()
    st.header("📚 Knowledge base")
    st.caption(f"Embeddings: `{kb.embedder.name}` · {len(kb.chunks)} chunks")
    uploads = st.file_uploader("Add documents", type=["pdf", "txt", "md", "csv", "docx"],
                               accept_multiple_files=True)
    if uploads and st.button("Ingest uploaded files", use_container_width=True):
        docs = []
        for f in uploads:
            data = f.getvalue()
            try:
                docs.append(load_document(f.name, data))
            except Exception as exc:
                st.error(f"{f.name}: {exc}")
            if f.name.lower().endswith(".csv"):
                toolbox.extra_tables[f.name] = data
        added = kb.add_documents(docs)
        kb.save(STORAGE_DIR)
        st.success(f"Added {added} document(s).")
        st.rerun()

    for d in kb.list_sources():
        st.markdown(f"- **{d['source']}** · {d['chunks']} chunks")

    c1, c2 = st.columns(2)
    if c1.button("Rebuild index", use_container_width=True):
        for p in STORAGE_DIR.glob("*"):
            p.unlink()
        get_kb.clear()
        st.session_state.pop("toolbox", None)
        st.rerun()
    if c2.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

agent = build_agent(toolbox, api_key=api_key or None, model=model, effort=effort)

# ------------------------------------------------------------------ main
st.title("🧠 Agentic RAG System")
mode = "Offline rule-based agent" if isinstance(agent, OfflineAgent) else f"Claude agent · `{model}`"
st.caption(f"Plans → decides when/how to retrieve → uses tools → answers with sources.  **Mode:** {mode}")

EXAMPLES = [
    "What is the difference between standard RAG and agentic RAG?",
    "How many annual leave days does an employee get after 6 years, and how many can be carried over?",
    "Compare what the documents say about hybrid search.",
    "What is the total revenue_usd by region in the sales data?",
    "DocuMind costs 1,200 USD/month. What would 12 months with Premium support cost?",
    "Summarize the employee handbook.",
]


def render_trace(steps):
    with st.expander(f"🔍 Agent trace ({len(steps)} steps)", expanded=False):
        for i, s in enumerate(steps, 1):
            if s.kind == "plan":
                st.markdown(f"**{i}. 🗺️ Plan**")
                st.json(s.input, expanded=True)
            elif s.kind == "thinking":
                st.markdown(f"**{i}. 💭 Reasoning**")
                st.caption(s.output[:1500])
            elif s.kind == "note":
                st.markdown(f"**{i}. ↻ {s.output}**")
            else:
                st.markdown(f"**{i}. 🛠️ `{s.name}`** `{json.dumps(s.input, ensure_ascii=False)}`")
                out = s.output
                if isinstance(out, dict) and "results" in out:
                    for r in out["results"]:
                        st.caption(f"{r['source']} · score {r['score']} — {r['text'][:220]}…")
                else:
                    st.code(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:2500], language="json")


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})"):
        for s in sources:
            score = f" · score {s['score']}" if s.get("score") is not None else ""
            st.markdown(f"**{s['source']}**{score}")
            st.caption(s["text"][:500])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if show_trace and m.get("steps"):
                render_trace(m["steps"])
            render_sources(m.get("sources"))

if not st.session_state.messages and "pending" not in st.session_state:
    st.markdown("**Try an example:**")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

question = st.chat_input("Ask a question about your documents…") or st.session_state.pop("pending", None)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Planning and retrieving…"):
            try:
                result = agent.answer(question, history)
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                st.stop()
        st.markdown(result.answer)
        if show_trace:
            render_trace(result.steps)
        render_sources(result.sources)
    st.session_state.messages.append({"role": "assistant", "content": result.answer,
                                      "steps": result.steps, "sources": result.sources})
