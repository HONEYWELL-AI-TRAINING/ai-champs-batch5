"""
What runs inside a node: an agent that answers (with optional tools) and a router that decides.

Both are async LangGraph nodes. Each one reports what it is doing through
`get_stream_writer()`, LangGraph's "custom" stream: the browser's progress tracking is nothing
more than those events, in the order they happen, from every node that is running, in parallel
or not.

    node_started   the node begins (which provider and model it will use)
    token          a piece of the answer, as the model writes it
    tool_call      the agent asked for a tool         tool_result   what the tool returned
    route          a router chose where to go next
    node_finished  the node's output, its time and its token counts
    node_error     the node failed (the run stops)

A node returns only a state update: its output, its place in the log and one step. LangGraph
merges parallel updates with the reducers in builder.RunState.
"""

from __future__ import annotations

import time
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field, create_model

from . import llm
from .schemas import FINISH, LENGTH_WORDS, END_ID, START_ID, NodeSpec, Workflow
from .tools import TOOLS, run_tool
from .validate import unique_edges

# An agent may call tools this many times before it must write its answer
MAX_TOOL_ROUNDS = 3
# What a tool result shows in the run panel; the model still receives all of it
PREVIEW_CHARS = 600

DEFAULT_AGENT_ROLE = "You are a capable, careful member of an IT-operations team."
DEFAULT_ROUTER_ROLE = (
    "You are the supervisor of a team of agents. Send the work to the agent best placed to move it forward, "
    "and finish as soon as the work so far answers the task."
)


# ── What an agent sees ────────────────────────────────────────────────────────

def upstream_ids(wf: Workflow, node_id: str) -> list[str]:
    """The agents wired into this one, in the order drawn (Start is not an agent)."""
    return [s for s, t in unique_edges(wf) if t == node_id and s != START_ID]


def _latest_in_order(state: dict, exclude: str) -> list[str]:
    """Node ids in the order they last finished, so a revised draft sits where the revision was made."""
    order: list[str] = []
    for node_id in state.get("log", []):
        if node_id in order:
            order.remove(node_id)
        order.append(node_id)
    return [i for i in order if i != exclude]


def build_context(spec: NodeSpec, wf: Workflow, state: dict) -> str:
    """The user message: the task, the work of other agents this node may see, and its own last attempt."""
    outputs: dict[str, str] = state.get("outputs", {})
    if spec.sees == "all":
        visible = _latest_in_order(state, spec.id)
    else:
        visible = [i for i in upstream_ids(wf, spec.id) if i in outputs]

    parts = [f"TASK:\n{state['task']}"]
    others = [f"### {wf.label_of(i)}\n{outputs[i]}" for i in visible if i in outputs]
    if others:
        parts.append("WORK FROM OTHER AGENTS:\n\n" + "\n\n".join(others))
    if spec.id in outputs and spec.type == "agent":
        parts.append("YOUR PREVIOUS OUTPUT (improve it; do not repeat it unchanged):\n" + outputs[spec.id])
    return "\n\n".join(parts)


def agent_system_prompt(spec: NodeSpec) -> str:
    words = LENGTH_WORDS[spec.length]
    lines = [
        (spec.role.strip() or DEFAULT_AGENT_ROLE),
        "",
        f"You are “{spec.label}”, one agent in a team working on a task together. Do your own part only: "
        f"the other agents cover the rest. Answer in at most {words} words, in plain text.",
    ]
    if spec.tools:
        lines.append("Use your tools when they help, and name the KB article or incident IDs you relied on.")
    return "\n".join(lines)


# ── Helpers shared by both kinds of node ──────────────────────────────────────

def _usage(message) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)


def _model_info(spec: NodeSpec) -> dict:
    provider = llm.get_provider(spec.provider or None)
    return {"provider": provider, "model": llm.model_name(provider, spec.tier)}


# ── An agent ──────────────────────────────────────────────────────────────────

def make_agent_node(spec: NodeSpec, wf: Workflow):
    async def agent(state: dict) -> dict:
        emit = get_stream_writer()
        t0 = time.perf_counter()
        try:
            info = _model_info(spec)
            emit({"event": "node_started", "node": spec.id, "label": spec.label, "type": "agent",
                  "step": state.get("steps", 0) + 1, **info})
            model = llm.get_chat_model(spec.provider or None, spec.tier)
            text, tokens_in, tokens_out = await _answer(spec, model, build_context(spec, wf, state), emit)
        except Exception as exc:
            emit({"event": "node_error", "node": spec.id, "error": llm.error_text(exc)})
            raise
        emit({
            "event": "node_finished", "node": spec.id, "output": text,
            "ms": round((time.perf_counter() - t0) * 1000), "input_tokens": tokens_in, "output_tokens": tokens_out,
        })
        return {"outputs": {spec.id: text}, "log": [spec.id], "steps": 1}

    return agent


async def _answer(spec: NodeSpec, model, context: str, emit) -> tuple[str, int, int]:
    """Stream the model's answer, running the agent's tools for up to MAX_TOOL_ROUNDS rounds first."""
    tools = [TOOLS[name] for name in spec.tools]
    messages: list = [SystemMessage(agent_system_prompt(spec)), HumanMessage(context)]
    tokens_in = tokens_out = 0
    spoken: list[str] = []

    for round_no in range(MAX_TOOL_ROUNDS + 1):
        # After the last tool round the model must write its answer, so it gets no tools
        runner = model.bind_tools(tools) if tools and round_no < MAX_TOOL_ROUNDS else model
        full = None
        async for chunk in runner.astream(messages):
            full = chunk if full is None else full + chunk  # chunks add up to the whole message, tool calls included
            if chunk.text:
                emit({"event": "token", "node": spec.id, "delta": chunk.text})
        reply = full if full is not None else AIMessage(content="")
        used_in, used_out = _usage(reply)
        tokens_in, tokens_out = tokens_in + used_in, tokens_out + used_out
        spoken.append(reply.text)

        calls = reply.tool_calls
        if not calls or round_no == MAX_TOOL_ROUNDS:
            # The answer is the last thing the model said; if a tool-calling round was all it ever said, use that
            return (reply.text.strip() or "\n".join(s for s in spoken if s).strip()), tokens_in, tokens_out

        messages.append(reply)
        for call in calls:
            emit({"event": "tool_call", "node": spec.id, "name": call["name"], "args": call["args"]})
            # Only the tools drawn on this agent run, whatever the model asks for
            granted = call["name"] in spec.tools
            result = run_tool(call["name"], call["args"]) if granted else f"Error: no tool called {call['name']}"
            emit({"event": "tool_result", "node": spec.id, "name": call["name"], "result": result[:PREVIEW_CHARS]})
            messages.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))
        emit({"event": "token", "node": spec.id, "delta": "\n\n"})  # a line between the rounds' text
    raise AssertionError("unreachable")  # the loop always returns on its last round


# ── A router ──────────────────────────────────────────────────────────────────

class Option(BaseModel):
    """One route a router can take: the name it answers with, where it leads and when to choose it."""

    name: str
    target: str  # a node id, or END_ID
    when: str


def route_options(spec: NodeSpec, wf: Workflow) -> list[Option]:
    """The routes leaving a router, from its outgoing edges, in the order drawn."""
    options = []
    edge_labels = {(e.source, e.target): e.label.strip() for e in wf.edges}
    for source, target in unique_edges(wf):
        if source != spec.id:
            continue
        label = edge_labels.get((source, target), "")
        if target == END_ID:
            options.append(Option(name=FINISH, target=END_ID, when=label or "the work so far answers the task"))
        else:
            node = wf.node(target)
            fallback = (node.role.strip().split(".")[0] if node else "")[:120]
            options.append(Option(name=node.label if node else target, target=target, when=label or fallback))
    return options


def route_schema(options: list[Option]) -> type[BaseModel]:
    """The structured answer a router must give: one of its own routes, why, and what to tell the next agent."""
    names = tuple(o.name for o in options)
    return create_model(
        "Route",
        next=(Literal[names], Field(description="The route to take next. Use one of the listed names exactly.")),  # type: ignore[valid-type]
        reason=(str, Field(description="One sentence: why this route.")),
        instruction=(str, Field(description="What the next agent should do, in one sentence. Empty when finishing.")),
    )


def router_system_prompt(spec: NodeSpec, options: list[Option]) -> str:
    lines = [
        (spec.role.strip() or DEFAULT_ROUTER_ROLE),
        "",
        f"You are “{spec.label}”, the router of the team. Decide the next step and use only these routes:",
    ]
    lines += [f"- {o.name}: {o.when}" for o in options]
    lines.append(
        f"Choose {FINISH} only when the work so far already answers the task. "
        "Do not send the work to an agent that has already done what is needed."
    )
    return "\n".join(lines)


def make_router_node(spec: NodeSpec, wf: Workflow):
    options = route_options(spec, wf)
    by_name = {o.name: o for o in options}
    schema = route_schema(options)
    graph_names = wf.slugs()  # Command(goto=…) takes the node's graph name, not the id the canvas uses

    async def router(state: dict) -> Command:
        emit = get_stream_writer()
        t0 = time.perf_counter()
        try:
            info = _model_info(spec)
            emit({"event": "node_started", "node": spec.id, "label": spec.label, "type": "router",
                  "step": state.get("steps", 0) + 1, **info})

            if state.get("steps", 0) + 2 >= wf.max_steps:
                # Sending the work on would take this router's step, the agent's, and still leave none for a router to
                # end the run with. So end it here, without paying for a decision: the limit is a ceiling
                nxt, target, forced = FINISH, END_ID, True
                reason = f"Step limit reached ({wf.max_steps}): the run ends here."
                instruction, tokens_in, tokens_out = "", 0, 0
            else:
                model = llm.get_chat_model(spec.provider or None, spec.tier)
                messages = [
                    SystemMessage(router_system_prompt(spec, options)),
                    HumanMessage(
                        build_context(spec, wf, state)
                        + f"\n\nSteps used so far: {state.get('steps', 0)} of {wf.max_steps}."
                    ),
                ]
                decision, tokens_in, tokens_out = await _decide(model, schema, messages)
                chosen = by_name[decision.next]
                nxt, target = chosen.name, chosen.target
                reason, instruction, forced = decision.reason.strip(), decision.instruction.strip(), False
        except Exception as exc:
            emit({"event": "node_error", "node": spec.id, "error": llm.error_text(exc)})
            raise

        emit({
            "event": "route", "node": spec.id, "next": target, "next_label": nxt, "reason": reason, "instruction": instruction, "forced": forced,
        })
        text = f"→ {nxt}. {reason}" + (f" Instruction: {instruction}" if instruction else "")
        emit({
            "event": "node_finished", "node": spec.id, "output": text,
            "ms": round((time.perf_counter() - t0) * 1000), "input_tokens": tokens_in, "output_tokens": tokens_out,
        })
        goto = END if target == END_ID else graph_names[target]
        return Command(update={"outputs": {spec.id: text}, "log": [spec.id], "steps": 1}, goto=goto)

    return router


async def _decide(model, schema: type[BaseModel], messages: list):
    """Ask for a structured route. A reply that does not parse is retried once; then the run stops with a clear error."""
    structured = model.with_structured_output(schema, include_raw=True)
    last_error = "no structured answer"
    for _ in range(2):
        result = await structured.ainvoke(messages)
        parsed = result.get("parsed")
        if parsed is not None:
            tokens_in, tokens_out = _usage(result.get("raw"))
            return parsed, tokens_in, tokens_out
        last_error = str(result.get("parsing_error") or last_error)
    raise ValueError(f"The router did not return a valid route: {last_error}")
