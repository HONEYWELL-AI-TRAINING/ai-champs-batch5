"""
The workflow the canvas draws, as plain data.

The browser sends this JSON to the server, which validates it, compiles it into a
LangGraph StateGraph and runs it. Nothing in it is code: an agent is a prompt, a model
choice and a few switches, and tools are picked from a fixed list on the server.

    {"nodes": [{"id": "n1", "type": "agent", "label": "Researcher", "role": "…"}, …],
     "edges": [{"source": "start", "target": "n1"}, {"source": "n1", "target": "end"}]}

"start" and "end" are the two fixed terminals of every workflow (LangGraph's START and END).
They are not in `nodes`; edges refer to them by these ids.
"""

from __future__ import annotations

import os
import re
from typing import Literal

from pydantic import BaseModel, Field

START_ID = "start"
END_ID = "end"
RESERVED_IDS = {START_ID, END_ID}
# What a router calls the edge that ends the run, and labels no agent may use
FINISH = "FINISH"
RESERVED_LABELS = {"start", "end", FINISH.lower()}

MAX_NODES = 12
# Fewer than 3 cannot hold: a router, the agent it picks and a closing step are already three
MIN_STEPS_LIMIT = 3
MAX_STEPS_LIMIT = 30
DEFAULT_MAX_STEPS = min(max(int(os.getenv("MAX_STEPS", "12") or 12), MIN_STEPS_LIMIT), MAX_STEPS_LIMIT)

NodeType = Literal["agent", "router"]
Tier = Literal["default", "small"]
Length = Literal["short", "medium", "long"]
Sees = Literal["upstream", "all"]
ToolName = Literal["search_kb", "lookup_incident", "calculator"]

# How long an agent's answer may be: the prompt says it, so the run stays quick and the demo readable
LENGTH_WORDS = {"short": 80, "medium": 180, "long": 400}


class NodeSpec(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    type: NodeType = "agent"
    label: str = Field(min_length=1, max_length=40)
    # An agent: its system prompt. A router: how it should decide.
    role: str = Field(default="", max_length=2000)
    # "" means LLM_PROVIDER
    provider: str = Field(default="", max_length=20)
    tier: Tier = "default"
    length: Length = "medium"
    # "upstream": the task plus the output of the agents wired into this one.
    # "all": the task plus everything produced so far.
    sees: Sees = "upstream"
    tools: list[ToolName] = Field(default_factory=list, max_length=3)
    # Canvas position. The server ignores it; it is kept so an exported file reopens as it was drawn.
    x: float = 0
    y: float = 0


class EdgeSpec(BaseModel):
    source: str
    target: str
    # On an edge that leaves a router: when to take it. Ignored elsewhere.
    label: str = Field(default="", max_length=160)


class Workflow(BaseModel):
    name: str = Field(default="Untitled workflow", max_length=80)
    nodes: list[NodeSpec] = Field(default_factory=list, max_length=MAX_NODES)
    edges: list[EdgeSpec] = Field(default_factory=list, max_length=MAX_NODES * MAX_NODES)
    # The most node runs one execution may use. A router that cannot fit another agent, and a closing step, ends the run
    max_steps: int = Field(default=DEFAULT_MAX_STEPS, ge=MIN_STEPS_LIMIT, le=MAX_STEPS_LIMIT)
    # Where Start and End sit on the canvas ({"start": {"x": 30, "y": 190}, …}); like x and y, kept for the editor only
    layout: dict[str, dict[str, float]] = Field(default_factory=dict)

    def node(self, node_id: str) -> NodeSpec | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def slugs(self) -> dict[str, str]:
        """A readable, unique LangGraph node name for every node id: "Network specialist" becomes network_specialist."""
        names: dict[str, str] = {}
        taken: set[str] = set()
        for node in self.nodes:
            base = re.sub(r"[^a-z0-9]+", "_", node.label.lower()).strip("_") or "node"
            if base[0].isdigit():
                base = f"n_{base}"
            name, i = base, 2
            while name in taken:
                name, i = f"{base}_{i}", i + 1
            taken.add(name)
            names[node.id] = name
        return names

    def label_of(self, node_id: str) -> str:
        if node_id == START_ID:
            return "Start"
        if node_id == END_ID:
            return FINISH
        node = self.node(node_id)
        return node.label if node else node_id


class RunRequest(BaseModel):
    workflow: Workflow
    task: str = Field(min_length=1, max_length=4000)
