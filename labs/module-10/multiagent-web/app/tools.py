"""
The tools an agent can be given on the canvas: a fixed list, all local, all read-only.

An agent with tools runs a small loop (the model asks, the tool answers, the model answers
again), which the run panel shows as tool calls. The data is the course's synthetic
IT-operations world (data/kb_articles.json, data/incidents.json): nothing here calls the network.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=None)
def _load(name: str) -> list[dict]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


@tool
def search_kb(query: str) -> str:
    """Search the IT knowledge base (KB-… articles: VPN, DNS, connection pools, access and more) and return the best matches."""
    wanted = _words(query)
    scored = []
    for article in _load("kb_articles.json"):
        # Title and tags count double: a hit there says more than a word buried in the body
        score = len(wanted & _words(article["body"])) + 2 * len(
            wanted & _words(article["title"] + " " + " ".join(article["tags"]))
        )
        if score:
            scored.append((score, article))
    scored.sort(key=lambda pair: -pair[0])
    if not scored:
        return "No knowledge-base article matched. Try different keywords."
    return "\n\n".join(
        f"{a['id']} · {a['title']} (team: {a['team']})\n{a['body']}" for _, a in scored[:2]
    )


@tool
def lookup_incident(incident_id: str) -> str:
    """Look up one incident by ID (for example INC-104233): summary, priority, region, owning team and status."""
    wanted = incident_id.strip().upper()
    for incident in _load("incidents.json"):
        if incident["id"] == wanted:
            return json.dumps({k: v for k, v in incident.items() if k != "expected_kb"}, indent=2)
    known = ", ".join(i["id"] for i in _load("incidents.json")[:4])
    return f"No incident {wanted}. Known IDs include {known}."


# +, -, *, / and ** on numbers, nothing else: an expression is parsed, never executed
_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("only numbers and + - * / ** are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(120 * 0.35) / 4'. Use it for any sum instead of doing it in your head."""
    try:
        result = _evaluate(ast.parse(expression.strip(), mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, SyntaxError, OverflowError) as exc:
        return f"Error: {exc}"
    return f"{result:,.6g}"


TOOLS = {t.name: t for t in (search_kb, lookup_incident, calculator)}

TOOL_INFO = [
    {"name": t.name, "description": (t.description or "").split(".")[0].strip()} for t in TOOLS.values()
]


def run_tool(name: str, args: dict) -> str:
    """Run one tool by name. An unknown name or bad arguments come back as text the model can read, not as a crash."""
    tool_fn = TOOLS.get(name)
    if tool_fn is None:
        return f"Error: no tool called {name}"
    try:
        return str(tool_fn.invoke(args))
    except Exception as exc:  # noqa: BLE001 — a tool failure is an observation for the model
        return f"Error: {exc}"
