"""
Internal HR Helper — An agent with tools to answer health insurance questions.
Uses LangGraph with Microsoft Foundry.
Ready for deployment to Foundry Hosted Agent service.

This module uses AzureAIResponsesAgentHost from a vendored copy of
https://github.com/langchain-ai/langchain-azure/pull/501 which provides
first-class LangGraph hosting support for Azure AI Foundry.

Run using:
    azd ai agent run
"""

import asyncio
import logging
import os
import re
from datetime import date

import httpx
import mcp.types
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from _vendor.langchain_azure_ai_runtime import AzureAIResponsesAgentHost

load_dotenv(override=True)

logger = logging.getLogger("hr-agent")
logger.setLevel(logging.INFO)

# Emit LangChain/LangGraph spans to Application Insights with gen_ai.agent.id
# so the Foundry portal Agent Monitor can identify this agent's traces.
enable_auto_tracing(
    auto_configure_azure_monitor=True,
    enable_content_recording=False,
    trace_all_langgraph_nodes=True,
    agent_id="hr-agent",
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


async def _build_agent():
    """Build the LangGraph agent with toolbox + local tools."""

    # The hosted platform auto-injects FOUNDRY_AGENT_TOOLBOX_ENDPOINT; fall back to
    # constructing it manually for local development.
    toolbox_endpoint = os.environ.get(
        "FOUNDRY_AGENT_TOOLBOX_ENDPOINT",
        f"{PROJECT_ENDPOINT.rstrip('/')}/toolboxes/{TOOLBOX_NAME}/mcp?api-version=v1",
    )
    extra_headers = {"Foundry-Features": TOOLBOX_FEATURES} if TOOLBOX_FEATURES else {}
    toolbox_client = MultiServerMCPClient(
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
        toolbox_tools = _sanitize_tools(await toolbox_client.get_tools())
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


# ── Hosted agent entrypoint ─────────────────────────────────────────


async def _main():
    """Build the agent and start the AzureAIResponsesAgentHost."""
    graph = await _build_agent()
    host = AzureAIResponsesAgentHost(
        graph=graph,
        stream_mode="messages",
        responses_history_count=20,
    )
    host.run()


if __name__ == "__main__":
    asyncio.run(_main())
