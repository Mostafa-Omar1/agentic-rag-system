"""Exercise the Claude agent loop with a fake client (no network, no API key)."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import ClaudeAgent, KnowledgeBase, Toolbox, load_directory  # noqa: E402
from rag.embeddings import TfidfEmbedder  # noqa: E402

DOCS = ROOT / "data" / "documents"


class FakeMessages:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.script.pop(0)


def tool_use(id_, name, inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def test_claude_agent_plans_searches_and_answers():
    kb = KnowledgeBase(embedder=TfidfEmbedder())
    kb.add_documents(load_directory(DOCS))
    toolbox = Toolbox(kb, DOCS)
    agent = ClaudeAgent.__new__(ClaudeAgent)
    import anthropic
    agent.anthropic, agent.model, agent.effort, agent.toolbox, agent._use_fallbacks = (
        anthropic, "claude-opus-5-5", "medium", toolbox, True)
    fake = FakeMessages([
        NS(stop_reason="tool_use", content=[
            NS(type="thinking", thinking="Need the leave policy."),
            tool_use("t1", "make_plan", {"question_type": "simple_lookup", "needs_retrieval": True,
                                          "steps": ["search leave policy"]})]),
        NS(stop_reason="tool_use", content=[
            tool_use("t2", "search_knowledge_base", {"query": "annual leave", "mode": "hybrid"}),
            tool_use("t3", "calculator", {"expression": "21 + 5"})]),
        NS(stop_reason="end_turn", content=[NS(type="text", text="21 days [helios_employee_handbook.txt]")]),
    ])
    agent.client = NS(beta=NS(messages=fake))

    result = agent.answer("How many leave days?")

    assert result.answer.startswith("21 days")
    assert [s.kind for s in result.steps] == ["thinking", "plan", "tool", "tool"]
    assert any(s["source"] == "helios_employee_handbook.txt" for s in result.sources)
    last = fake.calls[-1]
    assert last["fallbacks"] == "default" and last["thinking"]["type"] == "adaptive"
    # both tool results of the parallel call are returned in one user message
    assert [b["tool_use_id"] for b in last["messages"][-1]["content"]] == ["t2", "t3"]
