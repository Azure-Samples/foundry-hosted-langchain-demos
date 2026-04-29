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
from datetime import date

import httpx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_azure_ai.tools import AzureAIProjectToolbox
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from vendor.langchain_azure_ai_runtime import AzureAIResponsesAgentHost

load_dotenv(override=True)

logger = logging.getLogger("hr-agent")
logger.setLevel(logging.INFO)


# Emit LangChain/LangGraph spans to Application Insights with gen_ai.agent.id
# so the Foundry portal Agent Monitor can identify this agent's traces.
enable_auto_tracing(
    connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"],
    auto_configure_azure_monitor=True,
    enable_content_recording=True,
    trace_all_langgraph_nodes=True,
    agent_id="hr-agent",
)

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT_NAME = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]
TOOLBOX_NAME = os.environ.get("CUSTOM_FOUNDRY_AGENT_TOOLBOX_NAME", "hr-agent-tools")
SEARCH_ENDPOINT = os.environ["AZURE_AI_SEARCH_SERVICE_ENDPOINT"]
KB_NAME = os.environ.get("AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME", "zava-company-kb")

_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(_credential, "https://ai.azure.com/.default")
_search_token_provider = get_bearer_token_provider(_credential, "https://search.azure.com/.default")


class _AzureTokenAuth(httpx.Auth):
    """Attach a bearer token from a token provider to outgoing requests."""

    def __init__(self, provider):
        self._provider = provider

    def auth_flow(self, request):
        token = self._provider()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


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
    """Build the LangGraph agent with toolbox + KB MCP + local tools."""
    toolbox = AzureAIProjectToolbox(
        toolbox_name=TOOLBOX_NAME,
        credential=_credential,
    )

    toolbox_tools = await toolbox.get_tools()
    logger.info("Loaded %d toolbox tools", len(toolbox_tools))

    # Connect directly to Foundry IQ knowledge base via MCP
    mcp_url = (
        f"{SEARCH_ENDPOINT.rstrip('/')}/knowledgebases/{KB_NAME}"
        f"/mcp?api-version=2025-11-01-Preview"
    )
    kb_client = MultiServerMCPClient(
        {
            "knowledge-base": {
                "url": mcp_url,
                "transport": "streamable_http",
                "headers": {"Accept": "application/json, text/event-stream"},
                "auth": _AzureTokenAuth(_search_token_provider),
            }
        }
    )
    kb_tools = await kb_client.get_tools()
    logger.info("Loaded %d KB MCP tools", len(kb_tools))

    all_tools = [get_enrollment_deadline_info, get_current_date, *toolbox_tools, *kb_tools]

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
