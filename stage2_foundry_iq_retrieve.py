"""
Stage 2 (retrieve action): Add Foundry IQ grounding through the Azure AI Search
retrieve action instead of MCP.

This is a fallback when the KB MCP endpoint is not usable in your environment.

Prerequisites (in addition to Stage 1):
    AZURE_AI_SEARCH_SERVICE_ENDPOINT=https://<your-search>.search.windows.net
    AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME=zava-company-kb

Run:
    python stage2_foundry_iq_retrieve.py
"""

import asyncio
import logging
import os
from datetime import date
from typing import Annotated

import httpx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents.knowledgebases.aio import KnowledgeBaseRetrievalClient
from azure.search.documents.knowledgebases.models import (
    KnowledgeBaseRetrievalRequest,
    KnowledgeRetrievalSemanticIntent,
)
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import Field

load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("stage2-retrieve")
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

    kb_client = KnowledgeBaseRetrievalClient(
        endpoint=os.environ["AZURE_AI_SEARCH_SERVICE_ENDPOINT"],
        knowledge_base_name=os.environ["AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME"],
        credential=credential,
    )

    @tool
    async def knowledge_base_retrieve(
        queries: Annotated[
            list[str],
            Field(
                description=(
                    "1 to 3 concise search queries (max ~12 words each). "
                    "Use separate entries for alternate wording."
                ),
                min_length=1,
                max_length=3,
            ),
        ],
    ) -> str:
        """Search the Zava company knowledge base for HR policies and benefits."""
        logger.info("[tool] knowledge_base_retrieve(%s)", queries)
        request = KnowledgeBaseRetrievalRequest(
            intents=[KnowledgeRetrievalSemanticIntent(search=query) for query in queries]
        )
        result = await kb_client.retrieve(retrieval_request=request)
        if result.response and result.response[0].content:
            return result.response[0].content[0].text
        return "No results found."

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
    await kb_client.close()
    http_client.close()


if __name__ == "__main__":
    asyncio.run(main())
