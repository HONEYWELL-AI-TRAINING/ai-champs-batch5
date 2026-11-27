"""
Writes the LangGraph code that is equivalent to the workflow on the canvas.

This is the teaching view behind the editor's "Code" tab: the same nodes, the same edges and
the same joins that builder.py compiles, written the way you would write them by hand. It leaves
out what a demo needs and a first version does not: streaming, tools and the progress events.

The text is valid Python; tests/test_workflow.py runs it against a stub model to prove it.
"""

from __future__ import annotations

from .agents import DEFAULT_AGENT_ROLE, DEFAULT_ROUTER_ROLE, route_options, upstream_ids
from .builder import edge_plan
from .schemas import END_ID, FINISH, LENGTH_WORDS, START_ID, Workflow
from .validate import validate

HEADER = '''"""{title}

Generated from the canvas. Equivalent LangGraph code, simplified: no streaming, no tools, no progress events.
Needs: pip install langgraph langchain langchain-openai
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

model = init_chat_model("openai:gpt-5-mini")  # your provider and model
MAX_STEPS = {max_steps}  # the most nodes one run may use; a router that cannot fit another agent ends the run


class State(TypedDict):
    task: str
    outputs: Annotated[dict[str, str], lambda old, new: {{**old, **new}}]  # the latest answer per node
    steps: Annotated[int, operator.add]  # how many nodes have run


def ask(role: str, state: State, sees: list[str] | None = None) -> str:
    """One agent turn: the role is the system prompt; the task and the work it may see are the user message."""
    names = list(state["outputs"]) if sees is None else [n for n in sees if n in state["outputs"]]
    work = "\\n\\n".join(f"### {{n}}\\n{{state['outputs'][n]}}" for n in names)
    reply = model.invoke([("system", role), ("human", f"TASK:\\n{{state['task']}}\\n\\n{{work}}")])
    return reply.text
'''


def _literal(names: list[str]) -> str:
    return "Literal[" + ", ".join(repr(n) for n in names) + "]"


def generate_code(wf: Workflow) -> str:
    """Python source for the workflow, or comments saying why it cannot be written yet."""
    report = validate(wf)
    if not report.ok:
        return "# The workflow has problems, so there is no code yet:\n" + "\n".join(f"# - {e}" for e in report.errors) + "\n"

    names = wf.slugs()
    blocks = [HEADER.format(title=wf.name.replace('"', "'"), max_steps=wf.max_steps).rstrip()]

    def graph_name(node_id: str) -> str:
        return "START" if node_id == START_ID else "END" if node_id == END_ID else repr(names[node_id])

    for node in wf.nodes:
        name = names[node.id]
        if node.type == "agent":
            role = f"{node.role.strip() or DEFAULT_AGENT_ROLE} Answer in at most {LENGTH_WORDS[node.length]} words."
            sees = "None" if node.sees == "all" else repr([names[s] for s in upstream_ids(wf, node.id)])
            note = f"  # on the canvas this agent also has the tools: {', '.join(node.tools)}" if node.tools else ""
            blocks.append(
                f"def {name}(state: State) -> dict:{note}\n"
                f"    text = ask({role!r},\n"
                f"               state, sees={sees})\n"
                f"    return {{\"outputs\": {{{name!r}: text}}, \"steps\": 1}}"
            )
            continue

        options = route_options(node, wf)
        choices = [FINISH if o.target == END_ID else names[o.target] for o in options]
        route_class = f"{name.title().replace('_', '')}Route"
        prompt = (node.role.strip() or DEFAULT_ROUTER_ROLE) + "\nRoutes:\n" + "\n".join(
            f"- {c}: {o.when}" for c, o in zip(choices, options)
        )
        goto = "END if route.next == 'FINISH' else route.next" if FINISH in choices else "route.next"
        destinations = ["__end__" if c == FINISH else c for c in choices]  # the node names LangGraph can go to
        blocks.append(
            f"class {route_class}(BaseModel):\n"
            f"    next: {_literal(choices)}\n"
            f"    reason: str"
        )
        blocks.append(
            f"def {name}(state: State) -> Command[{_literal(destinations)}]:\n"
            f"    if state[\"steps\"] + 2 >= MAX_STEPS:  # no room for another agent plus a closing step: stop\n"
            f"        return Command(update={{\"steps\": 1}}, goto=END)\n"
            f"    work = \"\\n\\n\".join(f\"### {{n}}\\n{{t}}\" for n, t in state[\"outputs\"].items())\n"
            f"    route = model.with_structured_output({route_class}).invoke(\n"
            f"        [(\"system\", {prompt!r}),\n"
            f"         (\"human\", f\"TASK:\\n{{state['task']}}\\n\\n{{work}}\")]\n"
            f"    )\n"
            f"    return Command(update={{\"outputs\": {{{name!r}: route.reason}}, \"steps\": 1}},\n"
            f"                   goto={goto})"
        )

    assembly = ["builder = StateGraph(State)"]
    for node in wf.nodes:
        assembly.append(f"builder.add_node({names[node.id]!r}, {names[node.id]})")
    assembly.append("")
    for sources, target in edge_plan(wf, report.joins):
        if len(sources) > 1:
            assembly.append(
                f"builder.add_edge([{', '.join(graph_name(s) for s in sources)}], {graph_name(target)})"
                "  # a join: waits for every branch"
            )
        else:
            assembly.append(f"builder.add_edge({graph_name(sources[0])}, {graph_name(target)})")
    if any(n.type == "router" for n in wf.nodes):
        assembly.append("# A router has no add_edge: its return type, Command[Literal[...]], names where it can go")
    assembly.append("graph = builder.compile()")
    assembly.append("")
    assembly.append('# result = graph.invoke({"task": "…", "outputs": {}, "steps": 0})')
    assembly.append('# print(result["outputs"])')
    blocks.append("\n".join(assembly))
    return "\n\n\n".join(blocks) + "\n"
