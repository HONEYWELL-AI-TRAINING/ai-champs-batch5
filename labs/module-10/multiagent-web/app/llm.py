"""
LangChain chat models for the multi-agent canvas: any agent can use any configured provider.

LLM_PROVIDER is the default, and each agent on the canvas may pick another from the ones
that have their variables set. init_chat_model returns a ChatOpenAI or a ChatAnthropic; only
the model name, endpoint and key differ, so the agent code never changes.

    openai            OpenAI API                    OPENAI_API_KEY, OPENAI_MODEL
    anthropic         Claude API                    ANTHROPIC_API_KEY, ANTHROPIC_MODEL
    azure-openai      Azure OpenAI                  AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
    azure-anthropic   Claude on Microsoft Foundry   ANTHROPIC_FOUNDRY_API_KEY, ANTHROPIC_FOUNDRY_RESOURCE, ANTHROPIC_FOUNDRY_DEPLOYMENT

Each provider also has an optional "small" model (OPENAI_SMALL_MODEL, ANTHROPIC_SMALL_MODEL,
AZURE_OPENAI_SMALL_DEPLOYMENT, ANTHROPIC_FOUNDRY_SMALL_DEPLOYMENT) that an agent can pick to
answer faster and cheaper; without it, the small tier uses the main model.

Models are built on first use, so the app, /healthz and the tests run without a key.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

PROVIDERS = ("openai", "anthropic", "azure-openai", "azure-anthropic")

LABELS = {
    "openai": "OpenAI",
    "anthropic": "Claude",
    "azure-openai": "Azure OpenAI",
    "azure-anthropic": "Claude on Foundry",
}

# provider → (key variable, model variable, small-model variable, extra variable that must be set)
_ENV: dict[str, tuple[str, str, str, str | None]] = {
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_SMALL_MODEL", None),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_MODEL", None),
    "azure-openai": (
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_SMALL_DEPLOYMENT", "AZURE_OPENAI_ENDPOINT",
    ),
    "azure-anthropic": (
        "ANTHROPIC_FOUNDRY_API_KEY", "ANTHROPIC_FOUNDRY_DEPLOYMENT", "ANTHROPIC_FOUNDRY_SMALL_DEPLOYMENT",
        "ANTHROPIC_FOUNDRY_RESOURCE",
    ),
}

# Claude requires an output cap; an agent answers in a few paragraphs
CLAUDE_MAX_TOKENS = 2048
# Seconds one model call may take, so a stuck request ends the run with a clear error
REQUEST_TIMEOUT = 120


class ProviderConfigError(ValueError):
    """A provider is unknown, or a variable it needs is not set."""


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _require_env(name: str) -> str:
    value = _env(name)
    if not value:
        raise ProviderConfigError(f"{name} not found in environment variables!")
    return value


def get_provider(provider: str | None = None) -> str:
    """`provider`, else LLM_PROVIDER, else openai. Raises for a name that is not one of the four."""
    name = (provider or _env("LLM_PROVIDER") or "openai").lower()
    if name not in PROVIDERS:
        raise ProviderConfigError(f"Unsupported provider '{name}'. Use one of: {', '.join(PROVIDERS)}.")
    return name


def is_configured(provider: str) -> bool:
    key_var, model_var, _, extra_var = _ENV[provider]
    return all(_env(v) for v in (key_var, model_var, *([extra_var] if extra_var else [])))


def model_name(provider: str, tier: str = "default") -> str:
    """The model ID (the deployment name on Azure) for a provider and tier; small falls back to the main model."""
    _, model_var, small_var, _ = _ENV[provider]
    if tier == "small" and _env(small_var):
        return _env(small_var)
    return _require_env(model_var)


def _openai_options() -> dict:
    # stream_usage: the last streamed chunk carries the real token counts
    options: dict = {"stream_usage": True, "timeout": REQUEST_TIMEOUT}
    # Reasoning models think before the first token; a low effort shortens the wait
    if effort := _env("OPENAI_REASONING_EFFORT"):
        options["reasoning_effort"] = effort
    return options


@lru_cache(maxsize=16)
def get_chat_model(provider: str | None = None, tier: str = "default") -> BaseChatModel:
    """Build the chat model for a provider and tier on first use, then reuse it."""
    name = get_provider(provider)
    model = model_name(name, tier)
    key = _require_env(_ENV[name][0])

    if name == "openai":
        return init_chat_model(model, model_provider="openai", api_key=key, **_openai_options())
    if name == "anthropic":
        return init_chat_model(
            model, model_provider="anthropic", api_key=key, max_tokens=CLAUDE_MAX_TOKENS, timeout=REQUEST_TIMEOUT
        )
    if name == "azure-openai":
        # Azure OpenAI v1 API: the regular OpenAI client, base_url points at your Azure resource
        base_url = f"{_require_env('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/v1/"
        return init_chat_model(model, model_provider="openai", base_url=base_url, api_key=key, **_openai_options())
    # Claude on Microsoft Foundry: the Anthropic Messages API under /anthropic on your resource
    base_url = f"https://{_require_env('ANTHROPIC_FOUNDRY_RESOURCE')}.services.ai.azure.com/anthropic"
    return init_chat_model(
        model, model_provider="anthropic", base_url=base_url, api_key=key,
        max_tokens=CLAUDE_MAX_TOKENS, timeout=REQUEST_TIMEOUT,
    )


def describe() -> dict:
    """The default provider and every provider's state, for /healthz and the editor. Never shows a key."""
    try:
        default = get_provider()
        error = None
    except ProviderConfigError as exc:
        default, error = None, str(exc)

    providers = []
    for name in PROVIDERS:
        configured = is_configured(name)
        providers.append({
            "key": name,
            "label": LABELS[name],
            "configured": configured,
            "model": model_name(name) if configured else None,
            "small_model": model_name(name, "small") if configured else None,
        })
    info: dict = {"default_provider": default, "providers": providers}
    if error:
        info["config_error"] = error
    return info


# Anything shaped like an API key. Providers echo a masked copy of a wrong key in their 401 message; none of it is shown.
_KEY_LIKE = re.compile(r"\b(?:sk|key|api)[-_][A-Za-z0-9_\-*]{6,}", re.IGNORECASE)


def error_text(exc: Exception) -> str:
    """A short message for an `error` event: the provider's status and wording, never a stack trace or a key."""
    import anthropic
    import openai

    if isinstance(exc, ProviderConfigError):
        return str(exc)
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        if exc.status_code in (401, 403):
            return f"LLM provider error {exc.status_code}: the API key or permission was rejected. Check the key in .env."
        # 404: wrong model or deployment name · 429: rate limit
        return _KEY_LIKE.sub("[hidden]", f"LLM provider error {exc.status_code}: {exc.message}")
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return "Could not reach the LLM provider: check the network, endpoint or resource name"
    return _KEY_LIKE.sub("[hidden]", str(exc) or exc.__class__.__name__)
