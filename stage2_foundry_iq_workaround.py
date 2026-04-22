"""
Stage 2 (workaround): Add Foundry IQ grounding with a manual MCP wrapper.

This version bypasses the MCP adapter and manually calls the Azure AI Search
knowledge-base MCP endpoint, which is useful if the standard MCP client rejects
the response shape.

Prerequisites (in addition to Stage 1):
    AZURE_AI_SEARCH_SERVICE_ENDPOINT=https://<your-search>.search.windows.net
    AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME=zava-company-kb

Run:
    python stage2_foundry_iq_workaround.py
"""

import asyncio
import json
import logging
import os
from datetime import date
from typing import Annotated

import httpx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import Field

load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("stage2-workaround")
logger.setLevel(logging.INFO)


class _AzureTokenAuth(httpx.Auth):
    def __init__(self, provider):
        self._provider = provider

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self._provider()}"
        yield request


class KnowledgeBaseMCPTool:
    """Manual MCP wrapper for the Azure AI Search KB endpoint."""

    def __init__(self, http_client: httpx.Client, mcp_url: str) -> None:
        self._http_client = http_client
        self._mcp_url = mcp_url
        self._initialized = False
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._http_client.post(
            self._mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"sampling": {}},
                    "clientInfo": {"name": "stage2-agent", "version": "0.1.0"},
                },
            },
            headers=self._headers,
        ).raise_for_status()
        self._http_client.post(
            self._mcp_url,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=self._headers,
        ).raise_for_status()
        self._initialized = True

    def retrieve(self, queries: list[str]) -> str:
        self._ensure_initialized()
        response = self._http_client.post(
            self._mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "knowledge_base_retrieve",
                    "arguments": {"queries": queries},
                },
            },
            headers=self._headers,
        )
        response.raise_for_status()
        for line in response.text.splitlines():
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:].strip())
            result = data.get("result", {})
            content = result.get("content", [])
            snippets: list[str] = []
            for item in content:
                if item.get("type") == "resource" and "resource" in item:
                    snippets.append(item["resource"].get("text", ""))
                elif item.get("type") == "text":
                    snippets.append(item.get("text", ""))
            if snippets:
                return "\n\n---\n\n".join(snippets)
        return "No results found."


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
    search_http_client = httpx.Client(
        auth=_AzureTokenAuth(search_token_provider),
        timeout=httpx.Timeout(30.0, read=300.0),
    )
    kb_mcp_tool = KnowledgeBaseMCPTool(
        search_http_client,
        (
            f"{os.environ['AZURE_AI_SEARCH_SERVICE_ENDPOINT'].rstrip('/')}"
            f"/knowledgebases/{os.environ['AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME']}"
            f"/mcp?api-version=2025-11-01-Preview"
        ),
    )

    @tool
    def knowledge_base_retrieve(
        queries: Annotated[
            list[str],
            Field(
                description=(
                    "1 to 4 concise search queries (max ~12 words each). "
                    "Use alternate wording as separate entries."
                ),
                min_length=1,
                max_length=4,
            ),
        ],
    ) -> str:
        """Search the Zava company knowledge base for HR policies and benefits."""
        logger.info("[tool] knowledge_base_retrieve(%s)", queries)
        return kb_mcp_tool.retrieve(queries)

    agent = create_react_agent(
        llm,
        [get_enrollment_deadline_info, knowledge_base_retrieve],
        prompt=(
            f"You are an internal HR helper for Zava. Today's date is {date.today().isoformat()}. "
            "Use the knowledge-base retrieve tool to answer questions about HR policies, benefits, "
            "and company information. Use get_enrollment_deadline_info for enrollment timing."
        ),
    )

    result = await agent.ainvoke(
        {"messages": [("user", "What PerksPlus benefits are there, and when do I need to enroll by?")]}
    )
    print("\n--- Agent answer ---")
    print(_extract_assistant_text(result))
    search_http_client.close()
    http_client.close()


if __name__ == "__main__":
    asyncio.run(main())
