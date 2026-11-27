"""
Turns the drawn workflow into a compiled LangGraph StateGraph.

The mapping is small, and the "Code" tab in the editor shows it for the workflow on the canvas:

    an agent node                       graph.add_node(name, agent)
    a router node                       graph.add_node(name, router, destinations=(…))   (it returns Command(goto=…))
    an edge from an agent               graph.add_edge(source, target)
    two agents leaving one node         parallel branches: LangGraph runs both in the same superstep
    a node many edges lead into         graph.add_edge([a, b], node)   waits for all    (validate.join_modes: "all")
                                        or one add_edge per branch     runs on the first (validate.join_modes: "any")
    Start / End                         START / END
    node names                          the label, lower-cased: "Network specialist" becomes network_specialist

State is shared by every node. Each reducer says how parallel updates merge:

    outputs   the latest answer per node (dicts merge, a re-run replaces the old answer)
    log       node ids in the order they finished
    steps     how many nodes have run, which the routers compare with the workflow's step limit
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import make_agent_node, make_router_node
from .schemas import END_ID, START_ID, Workflow
from .validate import _adjacency, unique_edges, validate


def _merge(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class RunState(TypedDict):
    task: str
    outputs: Annotated[dict[str, str], _merge]
    log: Annotated[list[str], operator.add]
    steps: Annotated[int, operator.add]


class WorkflowError(ValueError):
    """The workflow has problems that stop it from compiling; `problems` lists them."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__(" ".join(problems))
        self.problems = problems


def edge_plan(wf: Workflow, joins: dict[str, str]) -> list[tuple[list[str], str]]:
    """The add_edge calls the graph needs, as (source ids, target id); several sources means "wait for all".

    A router has no entries here: it reaches its targets through Command(goto=…).
    """
    routers = {n.id for n in wf.nodes if n.type == "router"}
    _, backward = _adjacency(unique_edges(wf))
    plan: list[tuple[list[str], str]] = []
    for node in wf.nodes:
        sources = backward.get(node.id, [])
        if joins.get(node.id) == "all":
            plan.append((sources, node.id))
        else:
            plan.extend(([source], node.id) for source in sources if source not in routers)
    plan.extend(([source], END_ID) for source in backward.get(END_ID, []) if source not in routers)
    return plan


def build_graph(wf: Workflow, *, is_configured=None):
    """Validate and compile. Raises WorkflowError with every problem when the workflow cannot run."""
    report = validate(wf, is_configured=is_configured)
    if not report.ok:
        raise WorkflowError(report.errors)

    names = wf.slugs()
    forward, _ = _adjacency(unique_edges(wf))

    def graph_name(node_id: str) -> str:
        return START if node_id == START_ID else END if node_id == END_ID else names[node_id]

    graph = StateGraph(RunState)
    for node in wf.nodes:
        if node.type == "router":
            targets = tuple(graph_name(t) for t in forward[node.id])
            graph.add_node(names[node.id], make_router_node(node, wf), destinations=targets)
        else:
            graph.add_node(names[node.id], make_agent_node(node, wf))

    for sources, target in edge_plan(wf, report.joins):
        graph.add_edge(
            [graph_name(s) for s in sources] if len(sources) > 1 else graph_name(sources[0]), graph_name(target)
        )

    return graph.compile()


def mermaid(wf: Workflow) -> str:
    """The compiled graph as Mermaid text (dashed arrows are a router's possible routes)."""
    return build_graph(wf).get_graph().draw_mermaid()
