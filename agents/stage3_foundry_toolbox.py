"""
Stage 3: Add Foundry Toolbox — web search, code interpreter, and knowledge base via MCP.

What changes from Stage 2:
    - Replace the direct KB MCP tool with a Foundry Toolbox that bundles
      web_search, code_interpreter, and the Foundry IQ knowledge_base_retrieve
      tool behind a single MCP endpoint.

Prerequisites (in addition to Stage 1):
    - A Foundry Toolbox created with web_search, code_interpreter, and KB MCP tools.
      The azd up process uses "infra/create-toolbox.py" to create the toolbox.

Run:
    uv run python agents/stage3_foundry_toolbox.py
"""

import asyncio
import logging
import os
import re
from datetime import date

from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_azure_ai.tools import AzureAIProjectToolbox
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown

load_dotenv(override=True)

console = Console()
logger = logging.getLogger("stage3")


def _sanitize_tool_names(tools: list) -> list:
    """Fix MCP tool names for Responses API compatibility. Awaiting fix from Foundry Toolbox team."""
    for t in tools:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", t.name)
        if sanitized != t.name:
            t.name = sanitized
    return tools


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

        toolbox = AzureAIProjectToolbox(
            toolbox_name=os.environ["CUSTOM_FOUNDRY_AGENT_TOOLBOX_NAME"],
            credential=SyncDefaultAzureCredential(),
        )
        toolbox_tools = _sanitize_tool_names(await toolbox.get_tools())

        agent = create_agent(
            model=client,
            tools=[get_enrollment_deadline_info, *toolbox_tools],
            system_prompt=(
                f"You are an internal HR helper for Zava. Today's date is {date.today().isoformat()}. "
                "Use the knowledge base tool to answer questions about HR policies, benefits, "
                "and company information, and ground all answers in the retrieved context. "
                "Use get_enrollment_deadline_info for benefits enrollment timing. "
                "You can use web search to look up current information when the knowledge base "
                "does not have the answer. "
                "If you cannot answer from the tools, say so clearly."
            ),
        )

        result = await agent.ainvoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": "What PerksPlus benefits are available to employees?",
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
    logging.getLogger("mcp").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    asyncio.run(main())
