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

import mcp.types
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_azure_ai.tools import AzureAIProjectToolbox
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from vendor.langchain_azure_ai_runtime import AzureAIResponsesAgentHost

load_dotenv(override=True)

logger = logging.getLogger("hr-agent")
logger.setLevel(logging.INFO)


def _sanitize_tool_names(tools: list) -> list:
    """Fix MCP tool names for Responses API compatibility (^[a-zA-Z0-9_-]+$)."""
    for t in tools:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", t.name)
        if sanitized != t.name:
            logger.info("Renamed tool %r -> %r", t.name, sanitized)
            t.name = sanitized
    return tools

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

_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(_credential, "https://ai.azure.com/.default")


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


@tool
def get_current_date() -> str:
    """Return the current date in ISO format."""
    return date.today().isoformat()


@tool
def get_enrollment_deadline_info() -> dict:
    """Return enrollment timeline details for health insurance plans."""
    return {
        "enrollment_opens": "2026-11-11",
        "enrollment_closes": "2026-11-30",
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
    toolbox = AzureAIProjectToolbox(
        toolbox_name=TOOLBOX_NAME,
        credential=_credential,
    )

    toolbox_tools = []
    try:
        toolbox_tools = _sanitize_tool_names(await toolbox.get_tools())
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


def _main():
    """Build the agent and start the AzureAIResponsesAgentHost."""
    graph = asyncio.run(_build_agent())
    host = AzureAIResponsesAgentHost(
        graph=graph,
        stream_mode="messages",
        responses_history_count=20,
    )
    host.run()


if __name__ == "__main__":
    _main()
