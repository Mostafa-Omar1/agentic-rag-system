"""Tools the agent can call: search, read, summarize, compare, data analysis, calculation."""
from __future__ import annotations

import ast
import io
import json
import math
import operator
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .vector_store import KnowledgeBase, tokenize

# ------------------------------------------------------------------ calculator
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round,
    "min": min, "max": max, "sum": lambda *a: sum(a), "avg": lambda *a: sum(a) / len(a),
    "pct_change": lambda old, new: (new - old) / old * 100,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression without using eval()."""
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    expression = re.sub(r"(\d),(\d{3})", r"\1\2", expression)  # 1,000 -> 1000
    expression = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expression)

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = _ev(node.left), _ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            return _FUNCS[node.func.id](*[_ev(a) for a in node.args])
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"Unsupported expression element: {ast.dump(node)[:60]}")

    return _ev(ast.parse(expression, mode="eval"))


# ------------------------------------------------------------- summarization
def extractive_summary(text: str, max_sentences: int = 5, focus: str | None = None) -> str:
    """Frequency-based extractive summary (used when no LLM is configured)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 30]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    freq: dict[str, int] = {}
    for tok in tokenize(text):
        freq[tok] = freq.get(tok, 0) + 1
    focus_terms = set(tokenize(focus)) if focus else set()
    scored = []
    for i, s in enumerate(sentences):
        toks = tokenize(s)
        if not toks:
            continue
        score = sum(freq.get(t, 0) for t in toks) / len(toks)
        score += 3 * sum(1 for t in toks if t in focus_terms)
        score += 0.5 if i == 0 else 0
        scored.append((score, i, s))
    top = sorted(sorted(scored, reverse=True)[:max_sentences], key=lambda x: x[1])
    return " ".join(s for _, _, s in top)


# ----------------------------------------------------------------- toolbox
class Toolbox:
    """Holds the knowledge base and data directory; exposes tools by name."""

    def __init__(self, kb: KnowledgeBase, data_dir: str | Path, llm_summarizer=None):
        self.kb = kb
        self.data_dir = Path(data_dir)
        self.llm_summarizer = llm_summarizer  # callable(text, focus) -> str, optional
        self.extra_tables: dict[str, bytes] = {}  # uploaded CSVs

    # ---- knowledge base
    def search_knowledge_base(self, query: str, top_k: int = 5, mode: str = "hybrid",
                              sources: list[str] | None = None) -> dict:
        hits = self.kb.search(query, k=max(1, min(int(top_k), 15)), mode=mode, sources=sources or None)
        return {"query": query, "mode": mode, "results": [
            {"source": h["source"], "chunk_id": h["chunk_id"], "score": h["score"], "text": h["text"]}
            for h in hits]}

    def list_documents(self) -> dict:
        return {"documents": self.kb.list_sources(), "tables": self._table_names()}

    def read_document(self, source: str, max_chars: int = 6000) -> dict:
        text = self.kb.get_document_text(source)
        if text is None:
            return {"error": f"Document '{source}' not found. Use list_documents first."}
        return {"source": source, "truncated": len(text) > max_chars, "text": text[:max_chars]}

    def summarize(self, source: str | None = None, text: str | None = None, focus: str | None = None,
                  max_sentences: int = 5) -> dict:
        if source:
            text = self.kb.get_document_text(source)
            if text is None:
                return {"error": f"Document '{source}' not found."}
        if not text:
            return {"error": "Provide either 'source' or 'text'."}
        if self.llm_summarizer:
            try:
                return {"source": source, "summary": self.llm_summarizer(text, focus), "method": "llm"}
            except Exception as exc:  # fall back to extractive
                note = f"LLM summarizer failed ({exc}); used extractive."
                return {"source": source, "summary": extractive_summary(text, max_sentences, focus),
                        "method": "extractive", "note": note}
        return {"source": source, "summary": extractive_summary(text, max_sentences, focus),
                "method": "extractive"}

    def compare_sources(self, topic: str, sources: list[str] | None = None, top_k_per_source: int = 2) -> dict:
        names = sources or [d["source"] for d in self.kb.list_sources()]
        comparison = {}
        for name in names:
            hits = self.kb.search(topic, k=top_k_per_source, sources=[name])
            if hits:
                comparison[name] = [{"chunk_id": h["chunk_id"], "score": h["score"], "text": h["text"]}
                                    for h in hits]
        return {"topic": topic, "by_source": comparison}

    # ---- calculation
    def calculator(self, expression: str) -> dict:
        try:
            value = safe_eval(expression)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            return {"expression": expression, "result": value}
        except Exception as exc:
            return {"expression": expression, "error": str(exc)}

    # ---- data analysis
    def _table_names(self) -> list[str]:
        names = [p.name for p in self.data_dir.glob("*.csv")] if self.data_dir.exists() else []
        return sorted(set(names) | set(self.extra_tables))

    def _load_table(self, table: str) -> pd.DataFrame:
        if table in self.extra_tables:
            return pd.read_csv(io.BytesIO(self.extra_tables[table]))
        path = self.data_dir / Path(table).name
        if not path.exists():
            raise FileNotFoundError(f"Table '{table}' not found. Available: {self._table_names()}")
        return pd.read_csv(path)

    def analyze_data(self, table: str, operation: str = "describe", column: str | None = None,
                     group_by: str | None = None, filter_column: str | None = None,
                     filter_value: str | None = None, top_n: int = 10) -> dict:
        """operation: describe | head | sum | mean | median | min | max | count | value_counts | correlation"""
        try:
            df = self._load_table(table)
            if filter_column and filter_value is not None:
                if filter_column not in df.columns:
                    return {"error": f"Unknown column '{filter_column}'. Columns: {list(df.columns)}"}
                df = df[df[filter_column].astype(str).str.lower() == str(filter_value).lower()]
            for col in (column, group_by):
                if col and col not in df.columns:
                    return {"error": f"Unknown column '{col}'. Columns: {list(df.columns)}"}
            op = operation.lower()
            if op == "describe":
                result = json.loads(df.describe(include="all").fillna("").astype(str).to_json())
            elif op == "head":
                result = df.head(top_n).to_dict(orient="records")
            elif op == "value_counts":
                result = df[column].value_counts().head(top_n).to_dict()
            elif op == "correlation":
                result = json.loads(df.select_dtypes("number").corr().round(4).to_json())
            elif op in {"sum", "mean", "median", "min", "max", "count"}:
                if not column:
                    return {"error": f"'{op}' requires a column."}
                if group_by:
                    series = getattr(df.groupby(group_by)[column], op)()
                    series = series.sort_values(ascending=False).head(top_n)
                    result = {str(k): _py(v) for k, v in series.items()}
                else:
                    result = _py(getattr(df[column], op)())
            else:
                return {"error": f"Unknown operation '{operation}'."}
            return {"table": table, "operation": op, "column": column, "group_by": group_by,
                    "rows_used": int(len(df)), "columns": list(df.columns), "result": result}
        except Exception as exc:
            return {"error": str(exc)}

    # ---- dispatch
    def run(self, name: str, args: dict) -> dict:
        fn = getattr(self, name, None)
        if name.startswith("_") or name in {"run"} or not callable(fn):
            return {"error": f"Unknown tool '{name}'"}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}"}


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    return v


# JSON-schema definitions shared with the LLM agent.
TOOL_SCHEMAS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Retrieve the most relevant passages from the document knowledge base. Use 'hybrid' by default, "
            "'keyword' for exact names/numbers/codes, 'semantic' for paraphrased or conceptual questions. "
            "Restrict to specific files with 'sources'. Call multiple times with refined queries if results "
            "are weak or the question has several parts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused search query"},
                "top_k": {"type": "integer", "description": "Number of passages (1-15), default 5"},
                "mode": {"type": "string", "enum": ["hybrid", "semantic", "keyword"]},
                "sources": {"type": "array", "items": {"type": "string"},
                            "description": "Optional list of document filenames to search within"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_documents",
        "description": "List all documents and CSV data tables available in the knowledge base.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_document",
        "description": "Read the full (possibly truncated) text of one document by filename.",
        "input_schema": {
            "type": "object",
            "properties": {"source": {"type": "string"}, "max_chars": {"type": "integer"}},
            "required": ["source"],
        },
    },
    {
        "name": "summarize",
        "description": "Summarize a whole document (by 'source' filename) or a given 'text', optionally with a focus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"}, "text": {"type": "string"},
                "focus": {"type": "string", "description": "Aspect to emphasise"},
            },
        },
    },
    {
        "name": "compare_sources",
        "description": "Retrieve what each document says about a topic, grouped by document, for side-by-side comparison.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
                "top_k_per_source": {"type": "integer"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "analyze_data",
        "description": (
            "Run an analysis over a CSV table: describe, head, sum, mean, median, min, max, count, "
            "value_counts, correlation. Supports group_by and a single equality filter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "CSV filename, e.g. sales_2025.csv"},
                "operation": {"type": "string", "enum": ["describe", "head", "sum", "mean", "median", "min",
                                                          "max", "count", "value_counts", "correlation"]},
                "column": {"type": "string"},
                "group_by": {"type": "string"},
                "filter_column": {"type": "string"},
                "filter_value": {"type": "string"},
                "top_n": {"type": "integer"},
            },
            "required": ["table", "operation"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression exactly. Supports + - * / // % ** and sqrt, log, exp, abs, "
            "round, min, max, sum, avg, pct_change(old, new). Always use this instead of mental math."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]
