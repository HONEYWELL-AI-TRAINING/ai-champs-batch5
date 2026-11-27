"""
Multi-agent workflow canvas — FastAPI backend around a LangGraph that the browser draws.

The page lets you draw a workflow (agents, routers, the edges between them), configure each
node, and run it. The server never trusts the drawing as code: it is validated, compiled into a
LangGraph StateGraph (app/builder.py) and run, and every node reports its progress as it works.

Endpoints:
- GET  /api/config      → providers and their models, the tool list and the limits
- GET  /api/templates   → the ready-made workflows (sequential, parallel, supervisor, reflection)
- POST /api/validate    → problems in a drawn workflow, how its joins behave, and (when it is sound)
                          the compiled graph as Mermaid and the equivalent LangGraph code
- POST /api/run         → run a workflow on a task and stream its progress (SSE)
- GET  /healthz         → liveness probe, with the configured providers

Static files (the editor) are served from /static.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.errors import GraphRecursionError

from . import llm
from .builder import WorkflowError, build_graph, mermaid
from .codegen import generate_code
from .guardrails import check_input
from .schemas import (
    DEFAULT_MAX_STEPS, END_ID, LENGTH_WORDS, MAX_NODES, MAX_STEPS_LIMIT, MIN_STEPS_LIMIT, RunRequest, Workflow,
)
from .templates import TEMPLATES
from .tools import TOOL_INFO
from .validate import unique_edges, validate

app = FastAPI(title="Module 4 Multi-Agent Workflow Canvas")

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, **llm.describe()}


@app.get("/api/config")
def config() -> dict:
    return {
        **llm.describe(),
        "tools": TOOL_INFO,
        "lengths": LENGTH_WORDS,
        "limits": {"max_nodes": MAX_NODES, "min_steps": MIN_STEPS_LIMIT, "max_steps": MAX_STEPS_LIMIT,
                   "default_max_steps": DEFAULT_MAX_STEPS},
    }


@app.get("/api/templates")
def templates() -> list[dict]:
    return TEMPLATES


@app.post("/api/validate")
def validate_workflow(wf: Workflow) -> dict:
    """Everything the editor shows beside the canvas: problems, join behaviour, Mermaid and code."""
    report = validate(wf, is_configured=llm.is_configured)
    structure = validate(wf)  # the same checks without the providers, so Mermaid and code do not wait for .env
    body: dict = {
        "ok": report.ok,
        "errors": report.errors,
        "warnings": report.warnings,
        "joins": structure.joins,
        "mermaid": None,
        "code": None,
    }
    if structure.ok:
        body["mermaid"] = mermaid(wf)
        body["code"] = generate_code(wf)
    return body


@app.post("/api/run")
async def run(req: RunRequest) -> StreamingResponse:
    wf = req.workflow
    try:
        graph = build_graph(wf, is_configured=llm.is_configured)
    except WorkflowError as exc:
        # Nothing has been sent yet, so this can still be a proper HTTP error the editor lists
        raise HTTPException(400, detail={"message": "The workflow cannot run yet.", "problems": exc.problems})

    guard = check_input(req.task)

    async def event_stream():
        run_id = uuid.uuid4().hex[:8]
        yield _sse({"event": "run_started", "run_id": run_id, "max_steps": wf.max_steps, "task": guard["redacted"]})
        if not guard["ok"]:
            yield _sse({"event": "guardrail", "stage": "input", "reason": guard["reason"]})
            yield _sse({"event": "run_finished", "status": "blocked", "final": [], "path": [], "steps": 0,
                        "input_tokens": 0, "output_tokens": 0, "ms": 0})
            return

        t0 = time.perf_counter()
        tokens_in = tokens_out = 0
        forced = False
        final_state: dict = {}
        try:
            # custom: the progress events the nodes write; values: the whole state after each superstep
            async for mode, chunk in graph.astream(
                {"task": guard["redacted"], "outputs": {}, "log": [], "steps": 0},
                config={"recursion_limit": wf.max_steps * 2 + 10},
                stream_mode=["custom", "values"],
            ):
                if mode == "values":
                    final_state = chunk
                    continue
                if chunk.get("event") == "node_finished":
                    tokens_in += chunk.get("input_tokens", 0)
                    tokens_out += chunk.get("output_tokens", 0)
                elif chunk.get("event") == "route" and chunk.get("forced"):
                    forced = True
                yield _sse(chunk)
        except asyncio.CancelledError:
            # The browser closed the stream (Stop, or the tab went away): LangGraph cancels the running nodes, and
            # no further model calls are made
            print(f"[run] id={run_id} cancelled by the client", flush=True)
            raise
        except GraphRecursionError:
            yield _sse({"event": "error", "error": "The workflow looped past its safety limit and was stopped."})
            return
        except Exception as exc:  # a node already sent node_error; this ends the run
            yield _sse({"event": "error", "error": llm.error_text(exc)})
            return

        outputs, log = final_state.get("outputs", {}), final_state.get("log", [])
        print(f"[run] id={run_id} workflow={wf.name!r} steps={final_state.get('steps', 0)} "
              f"in={tokens_in} out={tokens_out} forced_stop={forced}", flush=True)
        yield _sse({
            "event": "run_finished",
            "status": "limit" if forced else "completed",
            "final": final_answer(wf, outputs, log),
            "path": log,
            "steps": final_state.get("steps", 0),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "ms": round((time.perf_counter() - t0) * 1000),
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Helpers ───────────────────────────────────────────────────────────────────

def final_answer(wf: Workflow, outputs: dict[str, str], log: list[str]) -> list[dict]:
    """The run's result: the agents wired to End that ran, else the last agent that ran (a router ended the run)."""
    def entry(node_id: str) -> dict:
        return {"node": node_id, "label": wf.label_of(node_id), "output": outputs[node_id]}

    ran_last = {nid: i for i, nid in enumerate(log)}
    last_to_end = [
        s for s, t in unique_edges(wf)
        if t == END_ID and (n := wf.node(s)) and n.type == "agent" and s in outputs
    ]
    if last_to_end:
        return [entry(i) for i in sorted(last_to_end, key=lambda i: ran_last.get(i, 0))]
    for node_id in reversed(log):
        node = wf.node(node_id)
        if node and node.type == "agent" and node_id in outputs:
            return [entry(node_id)]
    return []


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
