"""
Hosted HR helper built with LangGraph and Microsoft Foundry.

Uses the Responses protocol via ResponsesAgentServerHost for hosting.
The agent is built with ``create_agent`` (LangChain v1) which provides
a ReAct tool-calling loop. Conversation history is managed by the
platform via ``previous_response_id`` and ``context.get_history()``.

All tools — knowledge-base retrieval, web search, code interpreter —
come from a single Foundry Toolbox MCP endpoint, plus two local tools
for enrollment deadlines and the current date.

Run locally with:
    azd ai agent run
"""

import asyncio
import logging
import os
import re
from datetime import date

import httpx
import mcp.types
from azure.ai.agentserver.responses import (
    CreateResponse,
    ResponseContext,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
    TextResponse,
)
from azure.ai.agentserver.responses.models import (
    MessageContentInputTextContent,
    MessageContentOutputTextContent,
)
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

logger = logging.getLogger("hr-agent")
logger.setLevel(logging.INFO)

if not os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    logger.warning(
        "APPLICATIONINSIGHTS_CONNECTION_STRING not set — traces will not be sent to "
        "Application Insights. Set it for local telemetry; hosted containers inject it automatically."
    )

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT_NAME = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]
TOOLBOX_NAME = os.environ.get("CUSTOM_FOUNDRY_AGENT_TOOLBOX_NAME", "hr-agent-tools")
TOOLBOX_FEATURES = os.getenv("FOUNDRY_AGENT_TOOLBOX_FEATURES", "Toolboxes=V1Preview")

_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(_credential, "https://ai.azure.com/.default")


class _AzureTokenAuth(httpx.Auth):
    def __init__(self, provider):
        self._provider = provider

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self._provider()}"
        yield request

# Workaround: Azure AI Search KB MCP returns resource content with uri: null
# or uri: "", which fails pydantic AnyUrl validation in the MCP SDK.
for _cls in [
    mcp.types.ResourceContents,
    mcp.types.TextResourceContents,
    mcp.types.BlobResourceContents,
]:
    _cls.model_fields["uri"].annotation = str | None
    _cls.model_fields["uri"].default = None
    _cls.model_fields["uri"].metadata = []
for _cls in [
    mcp.types.ResourceContents,
    mcp.types.TextResourceContents,
    mcp.types.BlobResourceContents,
    mcp.types.EmbeddedResource,
    mcp.types.CallToolResult,
]:
    _cls.model_rebuild(force=True)


def _sanitize_tools(tools: list) -> list:
    """Fix MCP tool names/schemas for Responses API compatibility."""
    for tool_obj in tools:
        tool_obj.handle_tool_error = True
        # The Responses API requires tool names to match ^[a-zA-Z0-9_-]+$.
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_obj.name)
        if sanitized != tool_obj.name:
            logger.info("Renamed tool %r -> %r for Responses API compatibility", tool_obj.name, sanitized)
            tool_obj.name = sanitized
        schema = tool_obj.args_schema if isinstance(tool_obj.args_schema, dict) else None
        if schema is None:
            continue
        if schema.get("type") == "object" and "properties" not in schema:
            schema["properties"] = {}
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if required and not props:
            for field_name in required:
                props[field_name] = {"type": "string"}
            schema["properties"] = props
    return tools


@tool
def get_current_date() -> str:
    """Return the current date in ISO format."""
    return date.today().isoformat()


@tool
def get_enrollment_deadline_info() -> dict:
    """Return enrollment timeline details for health insurance plans."""
    return {
        "benefits_enrollment_opens": "2026-11-11",
        "benefits_enrollment_closes": "2026-11-30",
    }


SYSTEM_PROMPT = """You are an internal HR helper focused on employee benefits and company information.

Use the knowledge-base tool first for questions about Zava policies, benefits, plans,
deadlines, and internal company information.

Use toolbox tools such as web search when the knowledge base does not have the answer
or when the user asks for current external information.

Use get_enrollment_deadline_info and get_current_date when the question involves
benefits timing.

If the tools do not provide enough information, say so clearly and do not invent facts.
"""


# ── Agent setup ─────────────────────────────────────────────────────

_agent = None
_toolbox_client = None
_agent_lock = asyncio.Lock()


async def _build_agent():
    """Build the LangGraph agent with toolbox + local tools."""
    global _toolbox_client

    # The hosted platform auto-injects FOUNDRY_AGENT_TOOLBOX_ENDPOINT; fall back to
    # constructing it manually for local development.
    toolbox_endpoint = os.environ.get(
        "FOUNDRY_AGENT_TOOLBOX_ENDPOINT",
        f"{PROJECT_ENDPOINT.rstrip('/')}/toolboxes/{TOOLBOX_NAME}/mcp?api-version=v1",
    )
    extra_headers = {"Foundry-Features": TOOLBOX_FEATURES} if TOOLBOX_FEATURES else {}
    _toolbox_client = MultiServerMCPClient(
        {
            "toolbox": {
                "url": toolbox_endpoint,
                "transport": "streamable_http",
                "headers": extra_headers,
                "auth": _AzureTokenAuth(_token_provider),
            }
        }
    )

    toolbox_tools = []
    try:
        toolbox_tools = _sanitize_tools(await _toolbox_client.get_tools())
        logger.info("Loaded %d toolbox tools", len(toolbox_tools))
    except Exception as exc:
        logger.warning("Toolbox startup skipped: %s", exc)

    all_tools = [get_enrollment_deadline_info, get_current_date, *toolbox_tools]

    llm = ChatOpenAI(
        base_url=f"{PROJECT_ENDPOINT.rstrip('/')}/openai/v1",
        api_key=_token_provider,
        model=MODEL_DEPLOYMENT_NAME,
        use_responses_api=True,
        streaming=True,
    )

    return create_agent(model=llm, tools=all_tools, system_prompt=SYSTEM_PROMPT)


async def _get_agent():
    global _agent
    if _agent is not None:
        return _agent
    async with _agent_lock:
        if _agent is None:
            _agent = await _build_agent()
    return _agent


# ── Responses protocol handler ──────────────────────────────────────


def _history_to_langchain_messages(history: list) -> list:
    """Convert responses-protocol history items to LangChain messages."""
    messages = []
    for item in history:
        if hasattr(item, "content") and item.content:
            for content in item.content:
                if isinstance(content, MessageContentOutputTextContent) and content.text:
                    messages.append(AIMessage(content=content.text))
                elif isinstance(content, MessageContentInputTextContent) and content.text:
                    messages.append(HumanMessage(content=content.text))
    return messages


app = ResponsesAgentServerHost(
    options=ResponsesServerOptions(default_fetch_history_count=20)
)


@app.response_handler
async def handle_create(
    request: CreateResponse,
    context: ResponseContext,
    cancellation_signal: asyncio.Event,
):
    """Run the LangGraph agent and stream the response."""

    async def run_agent():
        try:
            agent = await _get_agent()
            try:
                history = await context.get_history()
            except Exception:
                history = []
            current_input = await context.get_input_text() or "Hello!"

            lc_messages = _history_to_langchain_messages(history)
            lc_messages.append(HumanMessage(content=current_input))

            result = await agent.ainvoke({"messages": lc_messages})

            # With use_responses_api, content may be a list of content blocks.
            raw = result["messages"][-1].content
            if isinstance(raw, list):
                yield "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in raw
                )
            else:
                yield raw or ""
        except Exception as exc:
            logger.exception("run_agent failed")
            yield f"[ERROR] {type(exc).__name__}: {exc}"

    return TextResponse(context, request, text=run_agent())


if __name__ == "__main__":
    app.run()
