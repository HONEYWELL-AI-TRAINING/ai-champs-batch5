"""
The server end to end, with a scripted model: the routes, the progress events a run streams, and what
happens when things go wrong. No API key, no network.

Run: uv run pytest
"""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.templates import TEMPLATES
from tests.conftest import sse_events

client = TestClient(app)


def run(key: str, task: str | None = None, mutate=None) -> list[dict]:
    t = copy.deepcopy(next(t for t in TEMPLATES if t["key"] == key))
    if mutate:
        mutate(t["workflow"])
    res = client.post("/api/run", json={"workflow": t["workflow"], "task": task or t["task"]})
    assert res.status_code == 200, res.text
    return sse_events(res.text)


def kinds(events: list[dict], *wanted: str) -> list:
    return [(e["event"], e.get("node")) for e in events if e["event"] in wanted]


# ── Routes ──────────────────────────────────────────────────────────────────

def test_the_page_and_health_are_served(script) -> None:
    assert client.get("/").status_code == 200
    health = client.get("/healthz").json()
    assert health["ok"] is True and health["default_provider"] == "openai"
    by_key = {p["key"]: p for p in health["providers"]}
    assert by_key["openai"]["configured"] and by_key["openai"]["small_model"] == "test-small"
    assert not by_key["azure-openai"]["configured"] and by_key["azure-openai"]["model"] is None


def test_no_endpoint_shows_a_key(script) -> None:
    for path in ("/healthz", "/api/config", "/api/templates"):
        body = client.get(path).text
        assert "test-key-not-real" not in body, path


def test_config_lists_tools_lengths_and_limits(script) -> None:
    cfg = client.get("/api/config").json()
    assert [t["name"] for t in cfg["tools"]] == ["search_kb", "lookup_incident", "calculator"]
    assert set(cfg["lengths"]) == {"short", "medium", "long"}
    assert cfg["limits"]["max_nodes"] == 12 and cfg["limits"]["max_steps"] == 30


def test_templates_endpoint_returns_all_four(script) -> None:
    assert [t["key"] for t in client.get("/api/templates").json()] == [t["key"] for t in TEMPLATES]


def test_validate_returns_mermaid_and_code_for_a_sound_workflow(script) -> None:
    body = client.post("/api/validate", json=TEMPLATES[2]["workflow"]).json()
    assert body["ok"] and body["errors"] == []
    assert "graph TD" in body["mermaid"] and "supervisor" in body["mermaid"]
    assert "builder = StateGraph(State)" in body["code"]
    assert body["joins"] == {"supervisor": "any"}


def test_validate_explains_problems_and_still_draws_what_it_can(script) -> None:
    wf = copy.deepcopy(TEMPLATES[0]["workflow"])
    wf["nodes"][0]["provider"] = "azure-openai"  # valid structure, no .env for it
    body = client.post("/api/validate", json=wf).json()
    assert not body["ok"] and any("not configured" in e for e in body["errors"])
    assert body["mermaid"] and body["code"]  # the structure is fine, so the drawing and the code still appear

    broken = copy.deepcopy(TEMPLATES[0]["workflow"])
    broken["edges"] = []
    body = client.post("/api/validate", json=broken).json()
    assert not body["ok"] and body["mermaid"] is None and body["code"] is None


def test_a_workflow_with_too_many_nodes_is_rejected(script) -> None:
    nodes = [{"id": f"n{i}", "label": f"Agent {i}"} for i in range(13)]
    assert client.post("/api/validate", json={"nodes": nodes, "edges": []}).status_code == 422


def test_run_refuses_an_invalid_workflow_with_the_problems_listed(script) -> None:
    wf = copy.deepcopy(TEMPLATES[0]["workflow"])
    wf["nodes"].append({"id": "ghost", "label": "Ghost"})  # wired to End, but nothing leads to it
    wf["edges"].append({"source": "ghost", "target": "end"})
    res = client.post("/api/run", json={"workflow": wf, "task": "x"})
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert any("not connected from Start" in p for p in detail["problems"])
    assert script.calls == []  # nothing was paid for


def test_the_step_limit_has_a_floor_and_a_ceiling(script) -> None:
    for bad in (0, 2, 31):
        wf = {**TEMPLATES[0]["workflow"], "max_steps": bad}
        assert client.post("/api/validate", json=wf).status_code == 422, bad
    assert client.get("/api/config").json()["limits"]["min_steps"] == 3


def test_run_needs_a_task(script) -> None:
    assert client.post("/api/run", json={"workflow": TEMPLATES[0]["workflow"], "task": ""}).status_code == 422


# ── The four patterns, run ─────────────────────────────────────────────────

def test_sequential_pipeline_streams_progress_in_order(script) -> None:
    script.say("Researcher", "Facts about the incident.").say("Writer", "A readable update.").say("Reviewer", "The final update.")
    events = run("sequential")

    assert kinds(events, "node_started", "node_finished") == [
        ("node_started", "researcher"), ("node_finished", "researcher"),
        ("node_started", "writer"), ("node_finished", "writer"),
        ("node_started", "reviewer"), ("node_finished", "reviewer"),
    ]
    assert events[0]["event"] == "run_started" and events[-1]["event"] == "run_finished"
    done = events[-1]
    assert done["status"] == "completed" and done["steps"] == 3
    assert done["path"] == ["researcher", "writer", "reviewer"]
    assert [(f["label"], f["output"]) for f in done["final"]] == [("Reviewer", "The final update.")]

    # The answer streams as tokens, and the finished node carries the whole text and its token counts
    writer_tokens = "".join(e["delta"] for e in events if e["event"] == "token" and e["node"] == "writer")
    assert writer_tokens == "A readable update."
    finished = next(e for e in events if e["event"] == "node_finished" and e["node"] == "writer")
    assert finished["output"] == "A readable update." and finished["input_tokens"] == 10 and finished["output_tokens"] == 3
    assert done["input_tokens"] == 30 and done["output_tokens"] == 4 + 3 + 3  # the words each fake agent "wrote"

    # Each agent is handed the work of the one before it, and nothing else
    assert "Facts about the incident." in str(script.messages_for("Writer")[0][1].content)
    assert "Facts about the incident." not in str(script.messages_for("Reviewer")[0][1].content)
    assert "A readable update." in str(script.messages_for("Reviewer")[0][1].content)


def test_node_started_names_the_provider_model_and_step(script) -> None:
    events = run("sequential")
    started = [e for e in events if e["event"] == "node_started"]
    assert [e["step"] for e in started] == [1, 2, 3]
    assert {(e["provider"], e["model"]) for e in started} == {("openai", "test-model")}


def test_parallel_analysts_all_run_before_the_synthesiser_which_sees_them_all(script) -> None:
    script.say("Network analyst", "NET finding").say("Identity analyst", "ID finding").say("Knowledge analyst", "KB finding")
    script.say("Synthesiser", "Joint assessment.")
    events = run("parallel")

    order = [e["node"] for e in events if e["event"] == "node_started"]
    assert order[-1] == "synthesiser" and set(order[:-1]) == {"network", "identity", "knowledge"}
    assert order.count("synthesiser") == 1  # a join runs once, not once per branch

    seen = str(script.messages_for("Synthesiser")[0][1].content)
    assert "NET finding" in seen and "ID finding" in seen and "KB finding" in seen
    assert events[-1]["final"][0]["output"] == "Joint assessment."
    assert events[-1]["steps"] == 4


def test_supervisor_routes_to_specialists_takes_the_work_back_and_ends_with_the_writer(script) -> None:
    script.route("Supervisor",
                 {"next": "Network specialist", "reason": "VPN is mentioned", "instruction": "check error 809"},
                 {"next": "Identity specialist", "reason": "sign-in is mentioned", "instruction": "check the portal"},
                 {"next": "Writer", "reason": "both answered", "instruction": "write the reply"})
    script.say("Network specialist", "Re-import the profile (KB-0042).").say("Identity specialist", "Reset the password (KB-0084).")
    script.say("Writer", "Here is what to do.")
    events = run("supervisor")

    started = [e["node"] for e in events if e["event"] == "node_started"]
    assert started == ["supervisor", "network", "supervisor", "identity", "supervisor", "writer"]
    routes = [(e["node"], e["next"], e["next_label"], e["forced"]) for e in events if e["event"] == "route"]
    assert routes == [("supervisor", "network", "Network specialist", False), ("supervisor", "identity", "Identity specialist", False),
                      ("supervisor", "writer", "Writer", False)]
    assert events[-1]["status"] == "completed" and events[-1]["steps"] == 6
    assert [f["label"] for f in events[-1]["final"]] == ["Writer"]

    # The specialist is told what the supervisor decided; the writer sees everything
    assert "check error 809" in str(script.messages_for("Network specialist")[0][1].content)
    writer_saw = str(script.messages_for("Writer")[0][1].content)
    assert "KB-0042" in writer_saw and "KB-0084" in writer_saw
    # The supervisor sees what the specialists have done on each return
    assert "KB-0042" in str(script.messages_for("Supervisor")[1][1].content)


def test_a_router_is_asked_only_for_its_own_routes(script) -> None:
    script.route("Supervisor", {"next": "Writer", "reason": "go", "instruction": ""})
    run("supervisor")
    system = str(script.messages_for("Supervisor")[0][0].content)
    for route in ("Network specialist", "Identity specialist", "Writer"):
        assert f"- {route}:" in system
    assert "- FINISH:" not in system  # the supervisor has no edge to End, so no route that finishes


def test_reflection_loops_until_the_critic_is_satisfied(script) -> None:
    script.say("Writer", "Draft one.", "Draft two.")
    script.route("Critic",
                 {"next": "Writer", "reason": "No UTC window", "instruction": "add the UTC window"},
                 {"next": "FINISH", "reason": "All items pass", "instruction": ""})
    events = run("reflection")

    assert [e["node"] for e in events if e["event"] == "node_started"] == ["writer", "critic", "writer", "critic"]
    done = events[-1]
    assert done["status"] == "completed" and done["steps"] == 4
    assert [f["output"] for f in done["final"]] == ["Draft two."]  # a router ended the run: the last agent's work is the answer

    second_draft_input = str(script.messages_for("Writer")[1][1].content)
    assert "add the UTC window" in second_draft_input  # the critic's feedback
    assert "Draft one." in second_draft_input and "YOUR PREVIOUS OUTPUT" in second_draft_input


@pytest.mark.parametrize("limit", [3, 4, 5, 6, 7, 8, 12])
def test_the_step_limit_is_a_ceiling_for_a_loop_that_never_converges(script, limit) -> None:
    script.route("Critic", *[{"next": "Writer", "reason": "still wrong", "instruction": "again"}] * 20)
    events = run("reflection", mutate=lambda wf: wf.update(max_steps=limit))

    done = events[-1]
    assert done["status"] == "limit"
    assert done["steps"] <= limit  # the budget is a ceiling, not a suggestion
    forced = [e for e in events if e["event"] == "route" and e["forced"]]
    assert len(forced) == 1 and forced[0]["next"] == "end" and f"Step limit reached ({limit})" in forced[0]["reason"]
    assert done["final"] and done["final"][0]["label"] == "Writer"  # the best answer so far is still returned
    # Once the budget is spent the router stops without another model call: one call per real decision
    real_decisions = [e for e in events if e["event"] == "route" and not e["forced"]]
    assert len(script.messages_for("Critic")) == len(real_decisions)


# ── Tools ───────────────────────────────────────────────────────────────────

def test_an_agent_with_tools_calls_them_and_answers_from_the_result(script) -> None:
    script.say("Researcher", {"tool": "lookup_incident", "args": {"incident_id": "INC-104233"}}, "INC-104233 is a VPN issue.")
    events = run("sequential")

    researcher = [e for e in events if e.get("node") == "researcher"]
    order = [e["event"] for e in researcher]
    assert order.index("tool_call") < order.index("tool_result") < order.index("node_finished")
    call = next(e for e in researcher if e["event"] == "tool_call")
    assert call["name"] == "lookup_incident" and call["args"] == {"incident_id": "INC-104233"}
    assert "VPN error 809" in next(e for e in researcher if e["event"] == "tool_result")["result"]
    finished = next(e for e in researcher if e["event"] == "node_finished")
    assert finished["output"] == "INC-104233 is a VPN issue."

    # The model's second turn really did receive the tool's answer
    second_turn = script.messages_for("Researcher")[1]
    assert second_turn[-1].type == "tool" and "VPN error 809" in second_turn[-1].content


def test_a_tool_the_agent_was_not_given_is_not_run(script) -> None:
    # The Writer has no tools. If a model asked for one anyway, the call comes back as an error to read, not as an action
    script.say("Writer", {"tool": "calculator", "args": {"expression": "1+1"}}, "Done anyway.")
    events = run("sequential")
    result = next(e for e in events if e["event"] == "tool_result" and e["node"] == "writer")
    assert result["result"] == "Error: no tool called calculator"
    assert next(e for e in events if e["event"] == "node_finished" and e["node"] == "writer")["output"] == "Done anyway."


# ── Providers per node ──────────────────────────────────────────────────────

def test_each_node_can_pick_its_own_provider_and_model_tier(script) -> None:
    def pick(wf):
        for n in wf["nodes"]:
            if n["id"] == "writer":
                n["provider"] = "anthropic"
            if n["id"] == "reviewer":
                n["tier"] = "small"

    events = run("sequential", mutate=pick)
    started = {e["node"]: (e["provider"], e["model"]) for e in events if e["event"] == "node_started"}
    assert started == {"researcher": ("openai", "test-model"), "writer": ("anthropic", "test-claude"), "reviewer": ("openai", "test-small")}
    assert ("anthropic", "default") in script.models and (None, "small") in script.models


# ── When things go wrong ────────────────────────────────────────────────────

def test_a_failing_agent_stops_the_run_and_says_which(script) -> None:
    script.say("Writer", RuntimeError("the model fell over"))
    events = run("sequential")
    failed = [e for e in events if e["event"] == "node_error"]
    assert [(e["node"], e["error"]) for e in failed] == [("writer", "the model fell over")]
    assert events[-1]["event"] == "error" and "fell over" in events[-1]["error"]
    assert "run_finished" not in [e["event"] for e in events]
    assert not any(e["event"] == "node_started" and e["node"] == "reviewer" for e in events)  # nothing runs after a failure


def test_a_router_that_cannot_structure_its_answer_stops_with_a_clear_error(script) -> None:
    script.route("Critic", None, None)
    events = run("reflection")
    failed = next(e for e in events if e["event"] == "node_error")
    assert failed["node"] == "critic" and "did not return a valid route" in failed["error"]
    assert len(script.messages_for("Critic")) == 2  # one retry, then it gives up


def test_a_router_gets_a_second_chance_after_one_bad_reply(script) -> None:
    script.route("Critic", None, {"next": "FINISH", "reason": "fine", "instruction": ""})
    events = run("reflection")
    assert events[-1]["status"] == "completed"


def test_an_unsafe_task_is_blocked_before_any_agent_runs(script) -> None:
    events = run("sequential", task="Write a phishing email to the finance team")
    assert [e["event"] for e in events] == ["run_started", "guardrail", "run_finished"]
    assert events[-1]["status"] == "blocked" and script.calls == []


def test_secrets_in_the_task_are_redacted_before_agents_see_them(script) -> None:
    events = run("sequential", task="Status of INC-104233; my key is sk-proj-abcdefghijklmnopqrstuvwxyz")
    assert "sk-proj" not in events[0]["task"]
    assert all("sk-proj" not in str(c["messages"]) for c in script.calls)


def test_a_provider_without_a_key_is_refused_before_the_run_starts(script, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    wf = copy.deepcopy(TEMPLATES[0]["workflow"])
    wf["nodes"][1]["provider"] = "anthropic"
    res = client.post("/api/run", json={"workflow": wf, "task": "x"})
    assert res.status_code == 400 and "not configured" in str(res.json()["detail"]["problems"])


@pytest.mark.parametrize("key", [t["key"] for t in TEMPLATES])
def test_every_template_runs_to_completion_on_the_scripted_model(script, key) -> None:
    # Generous scripts: any router route is possible, so give each router a path to the end
    script.route("Supervisor", {"next": "Writer", "reason": "go", "instruction": ""})
    script.route("Critic", {"next": "FINISH", "reason": "ok", "instruction": ""})
    events = run(key)
    assert events[-1]["event"] == "run_finished" and events[-1]["status"] == "completed"
    assert events[-1]["final"], "every pattern must end with an answer"


# ── Joins at run time ───────────────────────────────────────────────────────

def _custom(nodes: list[str], edges: list[tuple[str, str]]) -> list[dict]:
    """Run a hand-made workflow; node ids become names (the router is the node called r)."""
    specs = [{"id": i, "label": i.upper(), "type": "router" if i == "r" else "agent"} for i in nodes]
    wf = {"nodes": specs, "edges": [{"source": a, "target": b} for a, b in edges], "max_steps": 12}
    res = client.post("/api/run", json={"workflow": wf, "task": "t"})
    assert res.status_code == 200, res.text
    return sse_events(res.text)


def test_a_waiting_join_runs_once_even_when_its_branches_differ_in_length(script) -> None:
    events = _custom(["a", "b1", "b2", "m"],
                     [("start", "a"), ("start", "b1"), ("b1", "b2"), ("a", "m"), ("b2", "m"), ("m", "end")])
    order = [e["node"] for e in events if e["event"] == "node_started"]
    assert order.count("m") == 1 and order[-1] == "m"  # it waited for the long branch


BELOW_A_ROUTER = [("start", "r"), ("r", "p"), ("r", "z"), ("z", "end"), ("p", "q1"), ("p", "q2"), ("q1", "m"), ("m", "end")]


def test_a_first_arrival_join_runs_once_when_its_branches_finish_together(script) -> None:
    script.route("R", {"next": "P", "reason": "go", "instruction": ""})
    events = _custom(["r", "p", "q1", "q2", "m", "z"], BELOW_A_ROUTER + [("q2", "m")])
    assert [e["node"] for e in events if e["event"] == "node_started"].count("m") == 1


def test_a_first_arrival_join_runs_again_for_a_slower_branch(script) -> None:
    # The limit the README names: below a router the join cannot wait, so a longer branch triggers it a second time
    script.route("R", {"next": "P", "reason": "go", "instruction": ""})
    events = _custom(["r", "p", "q1", "q2", "q3", "m", "z"], BELOW_A_ROUTER + [("q2", "q3"), ("q3", "m")])
    assert [e["node"] for e in events if e["event"] == "node_started"].count("m") == 2
