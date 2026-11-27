"""
Shared test set-up: a scripted stand-in for the chat model, so every test runs offline.

The agents only ever call three things on a model: `bind_tools`, `astream` and
`with_structured_output(...).ainvoke`. FakeModel implements exactly those, replying from a
Script keyed by the agent's name (the system prompt says "You are “Name”"), so the order in which
parallel agents happen to call it does not matter.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableLambda

from app import llm

NAME = re.compile(r"You are “([^”]+)”")
PROVIDER_VARIABLES = [
    "LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_SMALL_MODEL", "OPENAI_REASONING_EFFORT",
    "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_MODEL",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_SMALL_DEPLOYMENT",
    "ANTHROPIC_FOUNDRY_API_KEY", "ANTHROPIC_FOUNDRY_RESOURCE", "ANTHROPIC_FOUNDRY_DEPLOYMENT",
    "ANTHROPIC_FOUNDRY_SMALL_DEPLOYMENT",
]


class Script:
    """What each agent says, in order. A reply is text, {"tool": name, "args": {...}}, or an exception to raise."""

    def __init__(self) -> None:
        self.replies: dict[str, list] = defaultdict(list)
        self.routes: dict[str, list] = defaultdict(list)
        self.calls: list[dict] = []  # every model call: who, with what messages, on which provider and tier
        self.models: list[tuple] = []  # (provider, tier) asked of get_chat_model

    def say(self, who: str, *items) -> "Script":
        self.replies[who].extend(items)
        return self

    def route(self, who: str, *decisions) -> "Script":
        self.routes[who].extend(decisions)
        return self

    def messages_for(self, who: str) -> list[list]:
        return [c["messages"] for c in self.calls if c["who"] == who]


def _who(messages) -> str:
    match = NAME.search(str(messages[0].content))
    return match.group(1) if match else "?"


class FakeModel:
    def __init__(self, script: Script, provider: str | None, tier: str) -> None:
        self.script, self.provider, self.tier = script, provider, tier

    def bind_tools(self, tools, **_):
        return self

    async def astream(self, messages):
        who = _who(messages)
        self.script.calls.append({"who": who, "messages": list(messages), "provider": self.provider, "tier": self.tier})
        queue = self.script.replies[who]
        item = queue.pop(0) if queue else f"{who} says something."
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict) and "tool" in item:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{"name": item["tool"], "args": json.dumps(item["args"]), "id": f"call_{who}", "index": 0}],
            )
            return
        words = item.split(" ")
        for i, word in enumerate(words):
            last = i == len(words) - 1
            yield AIMessageChunk(
                content=word + ("" if last else " "),
                usage_metadata={"input_tokens": 10, "output_tokens": len(words), "total_tokens": 10 + len(words)} if last else None,
            )

    def with_structured_output(self, schema, include_raw=False, **_):
        async def decide(messages):
            who = _who(messages)
            self.script.calls.append({"who": who, "messages": list(messages), "provider": self.provider, "tier": self.tier})
            queue = self.script.routes[who]
            if not queue:
                raise AssertionError(f"no scripted route left for {who}")
            item = queue.pop(0)
            raw = AIMessage(content="", usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
            if item is None:  # a reply the model could not structure
                return {"raw": raw, "parsed": None, "parsing_error": ValueError("bad json")}
            return {"raw": raw, "parsed": schema(**item), "parsing_error": None}

        return RunnableLambda(lambda _: None, afunc=decide)


@pytest.fixture
def script(monkeypatch) -> Script:
    """Providers configured, models faked: nothing in a test touches the network or the real .env."""
    for variable in PROVIDER_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_SMALL_MODEL", "test-small")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real-2")
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-claude")

    s = Script()

    def factory(provider=None, tier="default"):
        s.models.append((provider, tier))
        return FakeModel(s, provider, tier)

    monkeypatch.setattr(llm, "get_chat_model", factory)
    return s


def sse_events(body: str) -> list[dict]:
    return [json.loads(part[6:]) for part in body.split("\n\n") if part.startswith("data: ")]
