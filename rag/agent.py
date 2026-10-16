"""Agentic RAG: an agent that plans, decides when/how to retrieve, uses tools, and cites sources.

Two implementations share the same interface (``answer(question, history) -> AgentResult``):

* ``ClaudeAgent``  – Claude (tool use) drives the loop: plan -> retrieve/refine -> analyse -> answer.
* ``OfflineAgent`` – a deterministic rule-based planner used when no API key is configured,
  so the system can still be demoed end-to-end without network access.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .tools import TOOL_SCHEMAS, Toolbox
from .vector_store import tokenize

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
MAX_STEPS = 12

PLAN_TOOL = {
    "name": "make_plan",
    "description": (
        "Call this FIRST, before any other tool. Record your analysis of the question and a short "
        "step-by-step plan: which sub-questions to answer, which retrieval strategy/tools to use, and why. "
        "You may call it again later if you need to revise the plan."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question_type": {"type": "string", "enum": [
                "simple_lookup", "multi_hop", "comparison", "summarization", "data_analysis",
                "calculation", "chit_chat", "out_of_scope"]},
            "needs_retrieval": {"type": "boolean"},
            "sub_questions": {"type": "array", "items": {"type": "string"}},
            "steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["question_type", "needs_retrieval", "steps"],
    },
}

SYSTEM_PROMPT = """You are an Agentic RAG assistant that answers questions from a private knowledge base of \
documents and CSV data tables.

How to work:
1. Always start by calling `make_plan`: classify the question, decide whether retrieval is needed, break \
complex questions into sub-questions, and pick tools. Greetings or questions about yourself need no retrieval.
2. Retrieve with `search_knowledge_base`. Pick the mode deliberately (keyword for exact names/numbers, \
semantic for concepts, hybrid otherwise). If results are weak, off-topic or incomplete, refine the query \
(synonyms, narrower terms, a specific source) and search again. Answer each sub-question with evidence.
3. Use `compare_sources` for comparisons across documents, `summarize` for summaries, `analyze_data` for \
questions about CSV tables, and `calculator` for any arithmetic — never compute numbers in your head.
4. Answer only from retrieved evidence and tool results. If the knowledge base does not contain the answer, \
say so plainly instead of guessing.
5. Cite sources inline as [filename] after the sentences they support, and end with a short "Sources" list. \
Reply in the same language the user writes in. Be concise and well structured (markdown)."""


@dataclass
class Step:
    kind: str  # plan | tool | thinking | note
    name: str
    input: dict | None = None
    output: dict | str | None = None


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    mode: str = "claude"


def _collect_sources(name: str, output: dict, bag: dict) -> None:
    if not isinstance(output, dict):
        return
    if name == "search_knowledge_base":
        for r in output.get("results", []):
            bag.setdefault(r["chunk_id"], {"source": r["source"], "chunk_id": r["chunk_id"],
                                           "score": r["score"], "text": r["text"]})
    elif name == "compare_sources":
        for src, hits in output.get("by_source", {}).items():
            for h in hits:
                bag.setdefault(h["chunk_id"], {"source": src, "chunk_id": h["chunk_id"],
                                               "score": h["score"], "text": h["text"]})
    elif name in {"read_document", "summarize"} and output.get("source"):
        key = f"doc:{output['source']}"
        bag.setdefault(key, {"source": output["source"], "chunk_id": key, "score": None,
                             "text": (output.get("text") or output.get("summary") or "")[:600]})
    elif name == "analyze_data" and output.get("table"):
        key = f"table:{output['table']}"
        bag.setdefault(key, {"source": output["table"], "chunk_id": key, "score": None,
                             "text": json.dumps(output.get("result"), ensure_ascii=False)[:600]})


# =============================================================== Claude agent
class ClaudeAgent:
    def __init__(self, toolbox: Toolbox, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 effort: str = "medium"):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.toolbox = toolbox
        self._use_fallbacks = True
        toolbox.llm_summarizer = self.summarize_text

    # Server-side refusal fallback ("default" routing). If the account/platform rejects the
    # beta, retry once without it and remember that for later calls.
    def _create(self, **kwargs):
        if self._use_fallbacks:
            try:
                return self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            except self.anthropic.BadRequestError as exc:
                if "fallback" not in str(exc).lower():
                    raise
                self._use_fallbacks = False
        return self.client.beta.messages.create(**kwargs)

    def summarize_text(self, text: str, focus: str | None = None) -> str:
        prompt = "Summarize the following document in 4-6 bullet points"
        prompt += f", focusing on: {focus}" if focus else ""
        response = self._create(
            model=self.model, max_tokens=4000, output_config={"effort": "low"},
            messages=[{"role": "user", "content": f"{prompt}.\n\n<document>\n{text[:60000]}\n</document>"}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def answer(self, question: str, history: list[dict] | None = None) -> AgentResult:
        messages = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-6:]]
        messages.append({"role": "user", "content": question})
        steps: list[Step] = []
        bag: dict = {}
        tools = [PLAN_TOOL] + TOOL_SCHEMAS

        for _ in range(MAX_STEPS):
            response = self._create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": self.effort},
                tools=tools,
                messages=messages,
            )
            for block in response.content:
                if block.type == "thinking" and getattr(block, "thinking", ""):
                    steps.append(Step("thinking", "reasoning", output=block.thinking))

            if response.stop_reason == "refusal":
                return AgentResult("The request was declined by the model's safety system.", steps,
                                   list(bag.values()))
            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                if response.stop_reason == "max_tokens":
                    text += "\n\n_(answer truncated: max tokens reached)_"
                return AgentResult(text or "(no answer)", steps, list(bag.values()))

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                args = tu.input if isinstance(tu.input, dict) else {}
                if tu.name == "make_plan":
                    steps.append(Step("plan", "make_plan", input=args))
                    output = {"status": "plan recorded — now execute it"}
                else:
                    output = self.toolbox.run(tu.name, args)
                    steps.append(Step("tool", tu.name, input=args, output=output))
                    _collect_sources(tu.name, output, bag)
                results.append({
                    "type": "tool_result", "tool_use_id": tu.id,
                    "content": json.dumps(output, ensure_ascii=False, default=str)[:20000],
                    "is_error": isinstance(output, dict) and "error" in output,
                })
            messages.append({"role": "user", "content": results})

        return AgentResult("I could not finish within the step limit. Try a narrower question.", steps,
                           list(bag.values()))


# =============================================================== offline agent
_MATH = re.compile(r"(?:(?:sqrt|log|abs|round)\s*\()?[\d.,]+\s*(?:[-+*/^×÷%]\s*\(?\s*[\d.,]+\s*\)?\s*)+")
_PERCENT_OF = re.compile(r"([\d.,]+)\s*%\s*(?:of|من)\s*([\d.,]+)")
_PRICE = re.compile(r"([\d][\d,]*(?:\.\d+)?)\s*(?:usd|\$|dollars?)", re.I)
_PERIODS = re.compile(r"(\d+)\s*(?:months?|years?|weeks?|days?|users?|units?|licen[cs]es?)", re.I)
_SURCHARGE = re.compile(r"extra\s+(\d+(?:\.\d+)?)\s*%", re.I)
_DATA_WORDS = {"average": "mean", "mean": "mean", "total": "sum", "sum": "sum", "median": "median",
               "maximum": "max", "highest": "max", "max": "max", "minimum": "min", "lowest": "min",
               "min": "min", "count": "count", "how many": "count", "متوسط": "mean", "إجمالي": "sum",
               "مجموع": "sum", "أعلى": "max", "أقل": "min"}


class OfflineAgent:
    """Rule-based planner + extractive answerer. No LLM required."""

    def __init__(self, toolbox: Toolbox):
        self.toolbox = toolbox

    def _plan(self, q: str) -> dict:
        ql = q.lower()
        tables = self.toolbox._table_names()
        plan = {"question_type": "simple_lookup", "needs_retrieval": True, "sub_questions": [], "steps": []}
        if re.fullmatch(r"\s*(hi|hello|hey|thanks|thank you|مرحبا|اهلا|السلام عليكم|شكرا)[!.\s]*", ql):
            plan.update(question_type="chit_chat", needs_retrieval=False, steps=["Reply directly"])
            return plan
        if m := _PERCENT_OF.search(ql):
            plan.update(question_type="calculation", needs_retrieval=False, steps=["Use calculator"],
                        expression=f"{m.group(1)} / 100 * {m.group(2)}")
            return plan
        leftover = re.sub(r"what is|what's|calculate|compute|how much is|equals?|احسب|كم|ناتج|يساوي", "",
                          _MATH.sub("", ql))
        if _MATH.search(q) and not re.search(r"[a-z؀-ۿ]{3,}", leftover):
            plan.update(question_type="calculation", needs_retrieval=False, steps=["Use calculator"])
            return plan
        if any(w in ql for w in ("summarize", "summary", "overview", "لخص", "ملخص")):
            plan.update(question_type="summarization", steps=["Find target document", "Summarize it"])
        elif m := re.search(r"(?:difference|differ|compare)\s+(?:between\s+)?(.+?)\s+(?:and|vs\.?|versus|with)\s+(.+?)[?.]*$", ql):
            a, b = m.group(1).strip(), m.group(2).strip()
            plan.update(question_type="comparison", sub_questions=[a, b],
                        steps=[f"Search: {a}", f"Search: {b}", "Contrast the evidence"])
        elif any(w in ql for w in ("compare", "versus", " vs ", "قارن", "الفرق")):
            plan.update(question_type="comparison",
                        steps=["Retrieve topic per document with compare_sources", "Contrast findings"])
        elif tables and any(w in ql for w in _DATA_WORDS) and self._match_table(ql, tables):
            plan.update(question_type="data_analysis", steps=["Pick table/column", "Run analyze_data"])
        parts = [p.strip() for p in re.split(r"\?|;|\band then\b|\balso\b|\band what\b|\bو(?=ما|هل|كم)", q)
                 if len(p.strip()) > 8]
        if len(parts) > 1 and plan["question_type"] == "simple_lookup":
            plan.update(question_type="multi_hop", sub_questions=parts,
                        steps=[f"Search: {p}" for p in parts] + ["Combine evidence"])
        elif not plan["steps"]:
            plan["steps"] = ["Hybrid search", "Refine with keyword search if weak", "Extract answer"]
        if plan["question_type"] in ("simple_lookup", "multi_hop") and _PRICE.search(q) and _PERIODS.search(q):
            plan["question_type"] = "lookup_then_calculate"
            plan["steps"] = plan["steps"][:-1] + ["Find pricing rules in the documents", "Use calculator"]
        return plan

    def _derive_calculation(self, question: str, hits: list[dict], steps: list[Step]) -> str | None:
        """Combine numbers from the question with rules found in the retrieved text (e.g. surcharges)."""
        price = float(_PRICE.search(question).group(1).replace(",", ""))
        periods = int(_PERIODS.search(question).group(1))
        q_terms = set(tokenize(question))
        expr, rule = f"{price:g} * {periods}", None
        for h in hits:
            for s in re.split(r"(?<=[.!?])\s+|\n+", h["text"]):
                if (m := _SURCHARGE.search(s)) and len(q_terms & set(tokenize(s))) >= 2:
                    expr, rule = f"{price:g} * (1 + {m.group(1)} / 100) * {periods}", (s.strip(" -*"), h["source"])
                    break
            if rule:
                break
        out = self.toolbox.calculator(expr)
        steps.append(Step("tool", "calculator", input={"expression": expr}, output=out))
        if "result" not in out:
            return None
        why = f' (applying: "{rule[0]}" [{rule[1]}])' if rule else ""
        return f"**Calculation:** `{expr}` = **{out['result']:,}**{why}"

    @staticmethod
    def _match_table(ql: str, tables: list[str]) -> str | None:
        for t in tables:
            stem_terms = set(tokenize(t.rsplit(".", 1)[0].replace("_", " ")))
            if stem_terms & set(tokenize(ql)) or re.search(r"(data|table|csv|dataset)|بيانات|جدول", ql):
                return t
        return None

    def _best_sentences(self, question: str, hits: list[dict], n: int = 4) -> list[tuple[str, str]]:
        q = set(tokenize(question))
        scored = []
        for h in hits:
            for s in re.split(r"(?<=[.!?])\s+|\n+", h["text"]):
                s = s.strip().lstrip("-* ").strip()
                if s.startswith("#") or s.endswith(":") or (s and s[0].islower()):
                    continue  # headings, list intros, mid-sentence overlap fragments
                toks = set(tokenize(s))
                if len(s) < 25 or not toks:
                    continue
                overlap = len(q & toks) / (len(q) or 1)
                scored.append((overlap + 0.2 * h["score"], s.strip(), h["source"]))
        seen, out = set(), []
        for _, s, src in sorted(scored, reverse=True):
            if s not in seen:
                seen.add(s)
                out.append((s, src))
            if len(out) == n:
                break
        return out

    def _search(self, query: str, steps: list[Step], bag: dict, **kw) -> list[dict]:
        out = self.toolbox.search_knowledge_base(query, **kw)
        steps.append(Step("tool", "search_knowledge_base", input={"query": query, **kw}, output=out))
        _collect_sources("search_knowledge_base", out, bag)
        return out["results"]

    def answer(self, question: str, history: list[dict] | None = None) -> AgentResult:
        steps: list[Step] = []
        bag: dict = {}
        plan = self._plan(question)
        steps.append(Step("plan", "make_plan", input=plan))
        qt = plan["question_type"]

        if qt == "chit_chat":
            return AgentResult("Hello! Ask me anything about the documents in the knowledge base.", steps,
                               mode="offline")

        if qt == "calculation":
            expr = plan.get("expression") or _MATH.search(question).group(0)
            out = self.toolbox.calculator(expr)
            steps.append(Step("tool", "calculator", input={"expression": expr}, output=out))
            return AgentResult(f"**{expr.strip()} = {out.get('result', out.get('error'))}**", steps, mode="offline")

        if qt == "summarization":
            docs = self.toolbox.kb.list_sources()
            target = next((d["source"] for d in docs if d["source"].rsplit(".", 1)[0].lower().replace("_", " ")
                           in question.lower().replace("_", " ")), None)
            if not target:
                hits = self._search(question, steps, bag, top_k=3)
                target = hits[0]["source"] if hits else None
            if target:
                out = self.toolbox.summarize(source=target, max_sentences=6)
                steps.append(Step("tool", "summarize", input={"source": target}, output=out))
                _collect_sources("summarize", out, bag)
                return AgentResult(f"**Summary of {target}:**\n\n{out['summary']}\n\n**Sources:** [{target}]",
                                   steps, list(bag.values()), mode="offline")

        if qt == "comparison" and not plan["sub_questions"]:
            out = self.toolbox.compare_sources(question)
            steps.append(Step("tool", "compare_sources", input={"topic": question}, output=out))
            _collect_sources("compare_sources", out, bag)
            lines = ["**What each document says:**"]
            top = max((h["score"] for hs in out["by_source"].values() for h in hs), default=0)
            for src, hits in out["by_source"].items():
                if hits[0]["score"] < 0.5 * top:  # document is not really about this topic
                    continue
                best = self._best_sentences(question, [{**h, "source": src} for h in hits], n=2)
                if best:
                    lines.append(f"- **{src}:** " + " ".join(s for s, _ in best))
            return AgentResult("\n".join(lines), steps, list(bag.values()), mode="offline")

        if qt == "data_analysis":
            ql = question.lower()
            table = self._match_table(ql, self.toolbox._table_names())
            op = next(v for k, v in _DATA_WORDS.items() if k in ql)
            df = self.toolbox._load_table(table)
            numeric = list(df.select_dtypes("number").columns)
            grp = next((c for c in df.columns if f"by {c.lower()}" in ql or f"per {c.lower()}" in ql), None)
            num_col = next((c for c in numeric if c.lower() in ql or c.lower().split("_")[0] in ql),
                           numeric[-1] if numeric else None)
            if num_col:
                args = {"table": table, "operation": op, "column": num_col, "group_by": grp}
                out = self.toolbox.analyze_data(**args)
                steps.append(Step("tool", "analyze_data", input=args, output=out))
                _collect_sources("analyze_data", out, bag)
                return AgentResult(f"**{op}({num_col})**{' by ' + grp if grp else ''} in `{table}`:\n\n"
                                   f"```\n{json.dumps(out.get('result'), indent=2, ensure_ascii=False)}\n```",
                                   steps, list(bag.values()), mode="offline")

        # lookup / multi-hop: search each sub-question, refine if weak
        queries = plan["sub_questions"] or [question]
        all_hits: list[dict] = []
        for q in queries:
            hits = self._search(q, steps, bag, top_k=4)
            if not hits or hits[0]["score"] < 0.35:
                steps.append(Step("note", "refine", output="Weak results — retrying with keyword search"))
                hits += self._search(q, steps, bag, top_k=4, mode="keyword")
            all_hits.extend(hits)
        if not all_hits:
            return AgentResult("I couldn't find information about that in the knowledge base.", steps, mode="offline")
        lines = []
        for q in queries:
            best = self._best_sentences(q, all_hits, n=3 if len(queries) == 1 else 2)
            if len(queries) > 1:
                lines.append(f"**{q.strip()}?**")
            lines += [f"- {s} [{src}]" for s, src in best]
        if qt == "lookup_then_calculate" and (calc := self._derive_calculation(question, all_hits, steps)):
            lines.insert(0, calc + "\n\n**Evidence:**")
        srcs = sorted({h["source"] for h in all_hits})
        lines.append("\n**Sources:** " + ", ".join(f"[{s}]" for s in srcs))
        lines.append("\n_Offline mode: extractive answer. Add an Anthropic API key for full reasoning._")
        return AgentResult("\n".join(lines), steps, list(bag.values()), mode="offline")


def build_agent(toolbox: Toolbox, api_key: str | None = None, model: str = DEFAULT_MODEL,
                effort: str = "medium"):
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if key:
        return ClaudeAgent(toolbox, api_key=key, model=model, effort=effort)
    toolbox.llm_summarizer = None
    return OfflineAgent(toolbox)
