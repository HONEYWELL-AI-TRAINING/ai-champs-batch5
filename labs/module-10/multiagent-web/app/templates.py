"""
Four ready-made workflows for the template picker, one per orchestration pattern in the Module 4 guide.

    sequential   a pipeline: each agent hands its work to the next
    parallel     a fan-out: independent analysts work at once, one agent combines them (a join)
    supervisor   a router sends the work to specialists, takes it back, and finishes with a writer (§4.2)
    reflection   a writer and a critic loop until the critic is satisfied (the router decides when to stop)

They are plain data in the same shape the editor sends to the server, so each one is also a
worked example of the workflow format (app/schemas.py). Everything in them is synthetic.
"""

from __future__ import annotations


def _agent(node_id, label, role, x, y, **extra) -> dict:
    return {"id": node_id, "type": "agent", "label": label, "role": role, "x": x, "y": y, **extra}


def _router(node_id, label, role, x, y, **extra) -> dict:
    return {"id": node_id, "type": "router", "label": label, "role": role, "x": x, "y": y,
            "tier": "small", "sees": "all", "length": "short", **extra}


def _edge(source, target, label="") -> dict:
    return {"source": source, "target": target, "label": label}


SEQUENTIAL = {
    "key": "sequential",
    "name": "Sequential pipeline",
    "pattern": "Pipeline",
    "description": "Researcher, writer, reviewer: each hands its work to the next.",
    "task": "Write a short status update about incident INC-104233 for a non-technical manager: "
            "what happened, who is affected and what happens next.",
    "workflow": {
        "name": "Incident briefing pipeline",
        "max_steps": 8,
        "layout": {"start": {"x": 24, "y": 105}, "end": {"x": 1050, "y": 105}},
        "nodes": [
            _agent("researcher", "Researcher",
                   "You are an incident researcher on an IT-operations team. Look up the incident and the "
                   "knowledge-base article that matches it, and report only facts you found, with the INC and KB IDs.",
                   170, 60, tools=["lookup_incident", "search_kb"]),
            _agent("writer", "Writer",
                   "You write status updates for non-technical managers. From the research, explain in plain English "
                   "what happened, who is affected, what the team is doing and what happens next. No jargon, "
                   "and nothing the research does not say.",
                   460, 60),
            _agent("reviewer", "Reviewer",
                   "You are the editor. Check the update against the research, fix anything the research does not "
                   "support, and return only the final update, ready to send.",
                   750, 60),
        ],
        "edges": [
            _edge("start", "researcher"), _edge("researcher", "writer"),
            _edge("writer", "reviewer"), _edge("reviewer", "end"),
        ],
    },
}

PARALLEL = {
    "key": "parallel",
    "name": "Parallel analysts",
    "pattern": "Fan-out and join",
    "description": "Three analysts work at the same time; the synthesiser waits for all three.",
    "task": "Staff in the Frankfurt office report that the VPN drops after laptops wake from sleep and that "
            "some people cannot sign in to the HR portal. Give the team lead one joint assessment.",
    "workflow": {
        "name": "Parallel analysts",
        "max_steps": 8,
        "layout": {"start": {"x": 24, "y": 195}, "end": {"x": 930, "y": 195}},
        "nodes": [
            _agent("network", "Network analyst",
                   "You analyse IT issues from the network angle: VPN, DNS, gateways and connectivity. "
                   "Give the likely cause and one concrete check.",
                   170, 10, length="short"),
            _agent("identity", "Identity analyst",
                   "You analyse IT issues from the identity and access angle: sign-in, MFA, tokens and group "
                   "membership. Give the likely cause and one concrete check.",
                   170, 170, length="short"),
            _agent("knowledge", "Knowledge analyst",
                   "You search the knowledge base and report which KB articles apply, and what their steps and "
                   "rollback say.",
                   170, 340, length="short", tools=["search_kb"]),
            _agent("synthesiser", "Synthesiser",
                   "You combine the analysts' findings into one assessment for the team lead: the most likely root "
                   "cause, the order to check things in, and which team should own it.",
                   580, 170),
        ],
        "edges": [
            _edge("start", "network"), _edge("start", "identity"), _edge("start", "knowledge"),
            _edge("network", "synthesiser"), _edge("identity", "synthesiser"), _edge("knowledge", "synthesiser"),
            _edge("synthesiser", "end"),
        ],
    },
}

SUPERVISOR = {
    "key": "supervisor",
    "name": "Supervisor and specialists",
    "pattern": "Supervisor (guide §4.2)",
    "description": "A router sends the ticket to specialists, takes it back, and hands over to the writer.",
    "task": "Ticket: since this morning my VPN shows error 809 whenever my laptop wakes up, "
            "and I cannot sign in to the HR portal either. What should I do?",
    "workflow": {
        "name": "Supervisor with specialists",
        "max_steps": 10,
        "layout": {"start": {"x": 24, "y": 195}, "end": {"x": 1210, "y": 195}},
        "nodes": [
            _router("supervisor", "Supervisor",
                    "You supervise a support team. Ask the network specialist and the identity specialist when the "
                    "ticket needs them, each at most once, then send the work to the writer.",
                    170, 160),
            _agent("network", "Network specialist",
                   "You are the network specialist: VPN, DNS, gateways and connectivity. Answer only the network part "
                   "of the ticket, and cite the KB article you used.",
                   520, 0, tools=["search_kb"], length="short"),
            _agent("identity", "Identity specialist",
                   "You are the identity and access specialist: sign-in, MFA, passwords and access requests. Answer "
                   "only the sign-in part of the ticket, and cite the KB article you used.",
                   520, 300, tools=["search_kb"], length="short"),
            _agent("writer", "Writer",
                   "You write the reply to the user from the specialists' findings: short, friendly, numbered steps, "
                   "with the KB IDs. Do not add advice the specialists did not give.",
                   880, 160, sees="all"),
        ],
        "edges": [
            _edge("start", "supervisor"),
            _edge("supervisor", "network",
                  "the ticket mentions VPN, DNS or connectivity and the network specialist has not answered yet"),
            _edge("supervisor", "identity",
                  "the ticket mentions sign-in, MFA, passwords or access and the identity specialist has not answered yet"),
            _edge("supervisor", "writer", "the relevant specialists have answered, or none is needed"),
            _edge("network", "supervisor"), _edge("identity", "supervisor"),
            _edge("writer", "end"),
        ],
    },
}

REFLECTION = {
    "key": "reflection",
    "name": "Writer and critic loop",
    "pattern": "Reflection",
    "description": "The critic holds a checklist the writer has not seen, and sends the draft back until it passes.",
    "task": "Write an announcement for employees about planned maintenance of the VPN gateways on Saturday "
            "between 02:00 and 04:00 UTC.",
    "workflow": {
        "name": "Writer and critic",
        "max_steps": 8,
        "layout": {"start": {"x": 24, "y": 105}, "end": {"x": 980, "y": 105}},
        "nodes": [
            _agent("writer", "Writer",
                   "You write maintenance announcements for employees. Be friendly and complete. "
                   "When you get feedback, fix exactly what it names.",
                   170, 60, length="short"),
            _router("critic", "Critic",
                    "You review the announcement against this checklist, which the writer has not seen: "
                    "(1) exactly three sentences, (2) every sentence under 20 words, (3) the time window is given in UTC, "
                    "(4) it says what people should do. If any item fails, send it back and name the failed items in the "
                    "instruction. Otherwise finish.",
                    500, 60),
        ],
        "edges": [
            _edge("start", "writer"), _edge("writer", "critic"),
            _edge("critic", "writer", "the draft fails a checklist item: name the failed items in the instruction"),
            _edge("critic", "end", "the draft passes every checklist item"),
        ],
    },
}

TEMPLATES = [SEQUENTIAL, PARALLEL, SUPERVISOR, REFLECTION]
