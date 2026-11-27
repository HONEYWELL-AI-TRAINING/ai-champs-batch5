"""
Checks a drawn workflow before anything is compiled or paid for, and works out how its joins behave.

The rules exist because a LangGraph that compiles can still do the wrong thing: a loop with no
router never ends, a node nothing leads to never runs, a join that waits for a branch the router
did not take never fires. The editor shows every message below next to the graph.

Joins (a node with more than one incoming edge) behave in one of two ways:

    all   waits until every incoming branch has finished (parallel agents, then one that combines them)
    any   runs as soon as one incoming branch arrives (a node a router can reach, or one inside a loop)

A join can only wait for all branches when every branch is certain to run. A node inside a loop,
or downstream of a router, cannot rely on that, so it is "any".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .llm import PROVIDERS
from .schemas import END_ID, RESERVED_IDS, RESERVED_LABELS, START_ID, Workflow


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # node id → "all" or "any", for each node that has more than one incoming edge
    joins: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def unique_edges(wf: Workflow) -> list[tuple[str, str]]:
    """Each (source, target) pair once, in the order drawn."""
    seen: dict[tuple[str, str], None] = {}
    for e in wf.edges:
        seen.setdefault((e.source, e.target), None)
    return list(seen)


def _reach(starts: list[str], adjacency: dict[str, list[str]]) -> set[str]:
    """Everything reachable from `starts` by following one or more edges."""
    seen: set[str] = set()
    stack = list(starts)
    while stack:
        for nxt in adjacency.get(stack.pop(), []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _adjacency(edges: list[tuple[str, str]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    forward: dict[str, list[str]] = {}
    backward: dict[str, list[str]] = {}
    for source, target in edges:
        forward.setdefault(source, []).append(target)
        backward.setdefault(target, []).append(source)
    return forward, backward


def cycles(wf: Workflow) -> list[set[str]]:
    """The groups of nodes that can reach themselves (the loops), each group once."""
    edges = [(s, t) for s, t in unique_edges(wf) if s not in RESERVED_IDS and t not in RESERVED_IDS]
    forward, _ = _adjacency(edges)
    reach = {n.id: _reach([n.id], forward) for n in wf.nodes}
    groups: list[set[str]] = []
    for node in wf.nodes:
        if node.id not in reach[node.id]:
            continue
        group = {m for m in reach[node.id] if node.id in reach[m]} | {node.id}
        if group not in groups:
            groups.append(group)
    return groups


def join_modes(wf: Workflow) -> dict[str, str]:
    """"all" or "any" for every node with more than one incoming edge (see the module docstring)."""
    edges = unique_edges(wf)
    _, backward = _adjacency(edges)
    routers = {n.id for n in wf.nodes if n.type == "router"}
    looped = set().union(*cycles(wf)) if cycles(wf) else set()
    modes: dict[str, str] = {}
    for node in wf.nodes:
        sources = backward.get(node.id, [])
        if len(sources) < 2:
            continue
        ancestors = _reach([node.id], backward)
        certain = node.id not in looped and not (ancestors & routers)
        modes[node.id] = "all" if certain else "any"
    return modes


def validate(wf: Workflow, *, is_configured: Callable[[str], bool] | None = None) -> Report:
    """Every problem found, in plain words. `is_configured` also checks that each agent's provider can run."""
    report = Report()
    err, warn = report.errors.append, report.warnings.append

    if not wf.nodes:
        err("Add at least one agent: the workflow is empty.")
        return report

    # ── Names ──
    ids = [n.id for n in wf.nodes]
    if len(set(ids)) != len(ids):
        err("Two nodes share an id. Reload the template or re-create the duplicate.")
    for n in wf.nodes:
        if n.id in RESERVED_IDS:
            err(f"“{n.id}” is reserved for the Start and End terminals.")
    seen_labels: dict[str, str] = {}
    for n in wf.nodes:
        label = n.label.strip()
        if not label:
            err("Every node needs a name.")
            continue
        key = label.lower()
        if key in RESERVED_LABELS:
            err(f"“{label}” is reserved: a router uses it for the route that ends the run. Rename the agent.")
        elif key in seen_labels:
            err(f"Two nodes are called “{label}”. A router picks agents by name, so each name must be unique.")
        seen_labels[key] = n.id

    # ── Providers ──
    for n in wf.nodes:
        if n.provider and n.provider not in PROVIDERS:
            err(f"“{n.label}” asks for the unknown provider “{n.provider}”.")
        elif is_configured is not None:
            from .llm import ProviderConfigError, get_provider

            try:
                provider = get_provider(n.provider or None)
            except ProviderConfigError as exc:
                err(f"“{n.label}”: {exc}")
                continue
            if not is_configured(provider):
                err(f"“{n.label}” uses {provider}, which is not configured in .env.")

    # ── Edges ──
    known = set(ids) | RESERVED_IDS
    edges = unique_edges(wf)
    for source, target in edges:
        if source not in known or target not in known:
            err(f"An edge points at a node that does not exist ({source} → {target}).")
        elif source == END_ID:
            err("End has no outgoing edges: nothing runs after it.")
        elif target == START_ID:
            err("Nothing can lead back to Start.")
        elif source == target:
            err(f"“{wf.label_of(source)}” is wired to itself. To revise its own work, loop it through a router.")
    if report.errors:
        return report

    forward, backward = _adjacency(edges)
    if not forward.get(START_ID):
        err("Connect Start to at least one agent: nothing would run.")
    for n in wf.nodes:
        if not forward.get(n.id):
            err(f"“{n.label}” has no outgoing edge. Connect it to another agent or to End.")
        elif n.type == "router" and len(forward[n.id]) < 2:
            err(f"The router “{n.label}” has one route, so it has nothing to decide. Add a second, or make it an agent.")
    if report.errors:
        return report

    # ── Reachability and loops ──
    reachable = _reach([START_ID], forward)
    for n in wf.nodes:
        if n.id not in reachable:
            err(f"“{n.label}” is not connected from Start, so it would never run.")
    routers = {n.id for n in wf.nodes if n.type == "router"}
    for group in cycles(wf):
        if not group & routers:
            names = ", ".join(f"“{wf.label_of(i)}”" for i in sorted(group))
            err(f"{names} loop back on themselves with no router in the loop, so it would never end. "
                "Make one of them a router: it decides when to stop.")
    can_finish = _reach([END_ID], backward) | {END_ID}
    for n in wf.nodes:
        if n.id in reachable and n.id not in can_finish:
            warn(f"“{n.label}” has no path to End: the run ends only when the step limit is reached.")

    report.joins = join_modes(wf)
    return report
