"""
Stage 1: Same agent, now backed by an Azure OpenAI / Foundry model deployment.

Only the model client changes — the tool-calling agent flow stays the same.

Prerequisites:
    - An Azure OpenAI or Foundry model deployment
    - `az login`
    - `.env` with:
        AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
        AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-5.2

Run:
    python stage1_foundry_model.py
"""

import asyncio
import logging
import os
from datetime import date

import httpx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("stage1")
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

    agent = create_react_agent(
        llm,
        [get_enrollment_deadline_info],
        prompt=(
            f"You are an internal HR helper. Today's date is {date.today().isoformat()}. "
            "Use the available tools to answer questions about benefits enrollment timing. "
            "Always ground your answers in tool results."
        ),
    )

    result = await agent.ainvoke(
        {"messages": [("user", "When does benefits enrollment open?")]}
    )
    print("\n--- Agent answer ---")
    print(_extract_assistant_text(result))
    http_client.close()


if __name__ == "__main__":
    asyncio.run(main())
