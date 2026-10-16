# 🧠 Agentic RAG System

An **Agentic Retrieval-Augmented Generation** system that understands complex questions, **plans before answering**, decides **when and how to retrieve**, uses **tools** (search, summarization, comparison, data analysis, calculation), and produces **source-cited answers** from a knowledge base. It ships with a **Streamlit** web app.

## Standard RAG vs Agentic RAG

| | Standard RAG | Agentic RAG (this project) |
|---|---|---|
| Flow | Fixed: embed → top-k → generate | Agent loop: plan → act → observe → refine → answer |
| Retrieval | Always once, with the original query | Decides if/when/how; can retry with refined queries |
| Strategy | One retriever | Semantic, keyword (BM25) or hybrid; per-source filtering; full-document reading |
| Complex questions | Single pass | Decomposes into sub-questions (multi-hop) |
| Tools | None | Search, summarize, compare, analyze data (CSV), calculator |
| Output | Answer | Answer + inline citations + full reasoning trace |

## Architecture

```
            ┌───────────────── Processing pipeline (rag/ingest.py) ─────────────────┐
documents → │ extract (PDF/DOCX/TXT/MD/CSV) → clean → chunk (paragraph/sentence-aware, │
            │ 800 chars, 150 overlap) → embed (MiniLM / TF-IDF) → store (storage/)  │
            └────────────────────────────────────────────────────────────────────────┘
                                         │
                       KnowledgeBase (rag/vector_store.py): dense + BM25 hybrid search
                                         │
 user question → Agent (rag/agent.py) ── make_plan ──► tools (rag/tools.py) ──► cited answer
                    ▲                                     │
                    └──────── observe results, refine ◄───┘
```

* **`rag/ingest.py`** – text extraction, cleaning (control chars, de-hyphenation, whitespace), structure-aware chunking with overlap.
* **`rag/embeddings.py`** – `sentence-transformers/all-MiniLM-L6-v2` (normalized vectors) with automatic TF-IDF fallback.
* **`rag/vector_store.py`** – knowledge base with semantic, BM25 keyword and **hybrid** (min-max fused) search, source filters, persistence.
* **`rag/tools.py`** – the agent's toolbox:
  * `search_knowledge_base` (mode: hybrid / semantic / keyword, `sources` filter)
  * `list_documents`, `read_document`
  * `summarize` (LLM or extractive)
  * `compare_sources` (evidence grouped per document)
  * `analyze_data` (pandas over CSV: describe, sum, mean, median, min, max, count, value_counts, correlation, group-by, filter)
  * `calculator` (safe AST evaluator – no `eval`)
* **`rag/agent.py`**
  * **`ClaudeAgent`** – Claude tool-use loop. The agent must call `make_plan` first (question type, whether retrieval is needed, sub-questions, steps), then retrieves/refines/analyzes and answers with `[filename]` citations. Uses adaptive thinking with summarized reasoning shown in the trace.
  * **`OfflineAgent`** – a deterministic rule-based planner (no API key needed): routes to calculator / data analysis / summarize / compare / multi-hop search, refines weak results with keyword search, and builds extractive cited answers.
* **`app.py`** – Streamlit UI: chat, example questions, agent trace (plan → tool calls → results), sources panel, document upload & ingestion, index rebuild.

## Quick start

```bash
git clone https://github.com/Mostafa-Omar1/agentic-rag-system.git
cd agentic-rag-system
pip install -r requirements.txt
python scripts/build_index.py      # optional – the app builds the index on first run
streamlit run app.py
```

Open http://localhost:8501.

### Enable the Claude agent (recommended)

Without a key the app runs the **offline agent**. For full reasoning, provide an Anthropic API key, either in the sidebar or as an environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
export CLAUDE_MODEL=claude-opus-5-5        # optional, default
```

## Knowledge base

`data/documents/` contains sample documents (the company "Helios Analytics" is fictional):

| File | Content |
|---|---|
| `rag_overview.md` | Standard vs Agentic RAG, trade-offs, evaluation |
| `chunking_and_embeddings.md` | Chunking strategies, embeddings, hybrid search |
| `helios_employee_handbook.txt` | HR policies: leave, remote work, budgets, reviews |
| `helios_products.md` | Product catalogue with prices and codes |
| `sales_2025.csv` | Monthly sales by region and product (used by `analyze_data`) |

Add your own files to that folder (then click **Rebuild index**) or upload them from the sidebar.

## Example questions

* *What is the difference between standard RAG and agentic RAG?* → plan + multi-hop retrieval
* *How many annual leave days does an employee get after 6 years, and how many can be carried over?* → multi-step reasoning over one policy
* *Compare what the documents say about hybrid search.* → `compare_sources`
* *What is the total revenue_usd by region in the sales data?* → `analyze_data` (sum, group-by)
* *DocuMind costs 1,200 USD/month. What would 12 months with Premium support cost?* → search price + support rule, then `calculator`
* *Summarize the employee handbook.* → `summarize`

## Tests

```bash
pytest -q
```

Tests cover cleaning, chunking, all three retrieval modes, source filters, tools, offline-agent routing, and the Claude agent loop (with a mocked client – no network).

## Project structure

```
├── app.py                      # Streamlit app
├── rag/
│   ├── ingest.py               # extraction, cleaning, chunking
│   ├── embeddings.py           # embedding backends
│   ├── vector_store.py         # hybrid knowledge base + persistence
│   ├── tools.py                # agent tools + JSON schemas
│   └── agent.py                # Claude agent + offline agent
├── scripts/build_index.py      # CLI to (re)build the index
├── data/documents/             # knowledge-base documents
├── tests/                      # pytest suite
└── requirements.txt
```
