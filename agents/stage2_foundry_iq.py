"""
Stage 2: Add Foundry IQ — ground answers in an enterprise knowledge base
served by Azure AI Search via its MCP endpoint.

Uses LangChain's MCP adapters to discover and call the knowledge-base tool
exposed by Azure AI Search, with bearer-token auth via ``httpx.Auth``.

What changes from Stage 1:
    - Open an MCP client pointing at the KB MCP endpoint.
    - Pass the discovered tools to the agent alongside the local tools.
    - Update the system prompt to prefer the KB.

Prerequisites (in addition to Stage 1):
    AZURE_AI_SEARCH_SERVICE_ENDPOINT=https://<your-search>.search.windows.net
    AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME=zava-company-kb

Run:
    uv run python agents/stage2_foundry_iq.py
"""

import asyncio
import logging
import os
from datetime import date

import httpx
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown

load_dotenv(override=True)

console = Console()
logger = logging.getLogger("stage2")


class AzureTokenAuth(httpx.Auth):
    def __init__(self, provider) -> None:
        self._provider = provider

    async def async_auth_flow(self, request):
        token = await self._provider()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


@tool
def get_enrollment_deadline_info() -> dict:
    """Return enrollment timeline details for health insurance plans."""
    logger.info("[tool] get_enrollment_deadline_info()")
    return {
        "enrollment_opens": "2026-11-11",
        "enrollment_closes": "2026-11-30",
    }


async def main() -> None:
    credential = DefaultAzureCredential()
    try:
        aoai_token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        client = ChatOpenAI(
            base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/v1/",
            api_key=aoai_token_provider,
            model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
            use_responses_api=True,
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
                    "auth": AzureTokenAuth(search_token_provider),
                }
            }
        )
        kb_tools = await kb_client.get_tools()

        agent = create_agent(
            model=client,
            tools=[get_enrollment_deadline_info, *kb_tools],
            system_prompt=(
                f"You are an internal HR helper for Zava. Today's date is {date.today().isoformat()}. "
                "Use the knowledge-base tool to answer questions about HR policies, benefits, "
                "and company information, and ground all answers in the retrieved context. "
                "Use get_enrollment_deadline_info for benefits enrollment timing. "
                "If you cannot answer from the tools, say so clearly."
            ),
        )

        result = await agent.ainvoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": "What PerksPlus benefits are there, and when do I need to enroll by?",
                        }
                    ]
                })
        response = result["messages"][-1]
        console.print("\n[bold]Agent answer:[/bold]")
        console.print(Markdown(response.text))
    finally:
        await credential.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.core").setLevel(logging.WARNING)
    asyncio.run(main())
    