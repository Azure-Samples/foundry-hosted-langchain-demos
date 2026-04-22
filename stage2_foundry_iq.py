"""
Stage 2: Add Foundry IQ grounding through the Azure AI Search MCP endpoint.

This LangGraph-first version uses LangChain's MCP adapters to discover and call
knowledge-base tools exposed by Azure AI Search.

If the KB MCP endpoint still returns unsupported payloads in your environment,
use `stage2_foundry_iq_workaround.py` or `stage2_foundry_iq_retrieve.py`.

Prerequisites (in addition to Stage 1):
    AZURE_AI_SEARCH_SERVICE_ENDPOINT=https://<your-search>.search.windows.net
    AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME=zava-company-kb

Run:
    python stage2_foundry_iq.py
"""

import asyncio
import logging
import os
from datetime import date

import httpx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("stage2")
logger.setLevel(logging.INFO)


class _AzureTokenAuth(httpx.Auth):
    def __init__(self, provider):
        self._provider = provider

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self._provider()}"
        yield request


@tool
def get_enrollment_deadline_info() -> dict:
    """Return enrollment timeline details for health insurance plans."""
    logger.info("[tool] get_enrollment_deadline_info()")
    return {
        "benefits_enrollment_opens": "2026-11-11",
        "benefits_enrollment_closes": "2026-11-30",
    }


def _extract_assistant_text(result: dict) -> str:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for msg in reversed(messages):
        if getattr(msg, "type", "") != "ai":
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts)
    return ""


def _sanitize_tools(tools: list) -> list:
    for tool_obj in tools:
        tool_obj.handle_tool_error = True
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


async def main() -> None:
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    http_client = httpx.Client(auth=_AzureTokenAuth(token_provider), timeout=120.0)

    llm = ChatOpenAI(
        base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/v1",
        api_key="placeholder",
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        http_client=http_client,
    )

    search_token_provider = get_bearer_token_provider(
        credential, "https://search.azure.com/.default"
    )
    mcp_url = (
        f"{os.environ['AZURE_AI_SEARCH_SERVICE_ENDPOINT'].rstrip('/')}"
        f"/knowledgebases/{os.environ['AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME']}"
        f"/mcp?api-version=2025-11-01-Preview"
    )
    kb_client = MultiServerMCPClient(
        {
            "knowledge-base": {
                "url": mcp_url,
                "transport": "streamable_http",
                "headers": {"Accept": "application/json, text/event-stream"},
                "auth": _AzureTokenAuth(search_token_provider),
            }
        }
    )
    kb_tools = _sanitize_tools(await kb_client.get_tools())

    agent = create_react_agent(
        llm,
        [get_enrollment_deadline_info, *kb_tools],
        prompt=(
            f"You are an internal HR helper for Zava. Today's date is {date.today().isoformat()}. "
            "Use the knowledge-base tools to answer questions about HR policies, benefits, "
            "and company information. Use get_enrollment_deadline_info for enrollment timing. "
            "If the tools do not answer the question, say so clearly."
        ),
    )

    result = await agent.ainvoke(
        {"messages": [("user", "What PerksPlus benefits are there, and when do I need to enroll by?")]}
    )
    print("\n--- Agent answer ---")
    print(_extract_assistant_text(result))
    http_client.close()


if __name__ == "__main__":
    asyncio.run(main())
