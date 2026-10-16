"""Offline tests: pipeline, retrieval, tools and the offline agent.  Run:  pytest -q"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import KnowledgeBase, OfflineAgent, Toolbox, chunk_documents, clean_text, load_directory  # noqa: E402
from rag.embeddings import TfidfEmbedder  # noqa: E402
from rag.tools import safe_eval  # noqa: E402

DOCS = ROOT / "data" / "documents"


@pytest.fixture(scope="module")
def toolbox():
    kb = KnowledgeBase(embedder=TfidfEmbedder())
    kb.add_documents(load_directory(DOCS))
    return Toolbox(kb, DOCS)


def test_clean_text():
    assert clean_text("hel-\nlo   world\r\n\n\n\nnext") == "hello world\n\nnext"


def test_chunking_respects_size():
    docs = load_directory(DOCS)
    chunks = chunk_documents(docs, chunk_size=400, overlap=80)
    assert len(chunks) > len(docs)
    assert all(len(c.text) <= 600 for c in chunks)


@pytest.mark.parametrize("mode", ["hybrid", "semantic", "keyword"])
def test_search_finds_leave_policy(toolbox, mode):
    hits = toolbox.search_knowledge_base("annual leave days carried over", mode=mode)["results"]
    assert hits and hits[0]["source"] == "helios_employee_handbook.txt"


def test_keyword_search_exact_code(toolbox):
    hits = toolbox.search_knowledge_base("DM-220", mode="keyword")["results"]
    assert hits[0]["source"] == "helios_products.md"


def test_source_filter(toolbox):
    hits = toolbox.search_knowledge_base("search", sources=["rag_overview.md"])["results"]
    assert {h["source"] for h in hits} == {"rag_overview.md"}


def test_calculator():
    assert safe_eval("1,200 * 12 * (1 + 20%)") == pytest.approx(17280)
    assert safe_eval("sqrt(16) + 2^3") == 12
    with pytest.raises(Exception):
        safe_eval("__import__('os').system('echo hi')")


def test_analyze_data(toolbox):
    out = toolbox.analyze_data("sales_2025.csv", "sum", column="revenue_usd", group_by="region")
    assert out["result"]["UAE"] == 27400
    assert toolbox.analyze_data("sales_2025.csv", "mean", column="nope")["error"]


def test_compare_and_summarize(toolbox):
    cmp = toolbox.compare_sources("hybrid search")
    assert len(cmp["by_source"]) >= 2
    assert toolbox.summarize(source="helios_employee_handbook.txt")["summary"]


@pytest.mark.parametrize("question,expected_tool", [
    ("What is 15 * 4 + 10", "calculator"),
    ("What is the total revenue_usd by region in the sales data?", "analyze_data"),
    ("Compare what the documents say about hybrid search", "compare_sources"),
    ("Summarize the employee handbook", "summarize"),
    ("How many sick leave days do employees get?", "search_knowledge_base"),
])
def test_offline_agent_routes_tools(toolbox, question, expected_tool):
    result = OfflineAgent(toolbox).answer(question)
    assert result.steps[0].kind == "plan"
    assert expected_tool in [s.name for s in result.steps]
    assert result.answer


def test_offline_agent_percent_of(toolbox):
    result = OfflineAgent(toolbox).answer("What is 15% of 2400?")
    assert "360" in result.answer


def test_offline_agent_retrieves_then_calculates(toolbox):
    q = "DocuMind costs 1,200 USD/month. What would 12 months with Premium support cost?"
    result = OfflineAgent(toolbox).answer(q)
    names = [s.name for s in result.steps]
    assert names.index("search_knowledge_base") < names.index("calculator")
    assert "17,280" in result.answer
