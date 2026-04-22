"""
Stage 0: Fully local LangGraph agent using a small model via Ollama.

No cloud, no Azure account required. Demonstrates the core tool-calling loop:
    user -> model -> tool call -> tool result -> final answer

Prerequisites:
    1. Install Ollama: https://ollama.com/download
    2. Pull a small tool-capable model, for example:
         ollama pull qwen3.5:4b
    3. Make sure Ollama is running on http://localhost:11434/v1

Run:
    python stage0_local_agent.py
"""

import asyncio
import logging
from datetime import date

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("stage0")
logger.setLevel(logging.INFO)


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
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3.5:4b",
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


if __name__ == "__main__":
    asyncio.run(main())
