"""
Workflow as Agent: Writer → Formatter pipeline with streaming output.

Demonstrates LangGraph's StateGraph with astream — each node produces
output that is printed as it arrives, showing the multi-step processing.

This is the local runnable version of the workflow pattern used by the
hosted version in stage4_foundry_hosted_as_agent.py.

Prerequisites:
    - An Azure OpenAI / Foundry model deployment
    - `az login` (uses DefaultAzureCredential)
    - .env with:
        AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
        AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-5.2

Run:
    python workflows/stage3_as_agent.py
"""

import asyncio
import os
from typing import TypedDict

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv(override=True)


class WorkflowState(TypedDict):
    content: str


async def main():
    """Build a Writer → Formatter workflow and stream results."""
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )

    llm = ChatOpenAI(
        base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/v1/",
        api_key=token_provider,
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        streaming=True,
    )

    async def writer(state: WorkflowState) -> WorkflowState:
        response = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a concise content writer. "
                        "Write a clear, engaging short article (2-3 paragraphs) based on the user's topic. "
                        "Focus on accuracy and readability."
                    ),
                },
                {"role": "user", "content": state["content"]},
            ]
        )
        return {"content": response.content}

    async def formatter(state: WorkflowState) -> WorkflowState:
        response = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an expert content formatter. "
                        "Take the provided text and format it with Markdown (bold, headers, lists) "
                        "and relevant emojis to make it visually engaging. "
                        "Preserve the original meaning and content."
                    ),
                },
                {"role": "user", "content": state["content"]},
            ]
        )
        return {"content": response.content}

    graph = StateGraph(WorkflowState)
    graph.add_node("writer", writer)
    graph.add_node("formatter", formatter)
    graph.add_edge(START, "writer")
    graph.add_edge("writer", "formatter")
    graph.add_edge("formatter", END)

    workflow = graph.compile()

    prompt = "Write a short post about why open-source AI frameworks matter."
    print(f"Prompt: {prompt}\n")
    print("=" * 60)

    async for event in workflow.astream({"content": prompt}, stream_mode="updates"):
        for node_name, update in event.items():
            print(f"\n[{node_name}]:")
            print(update.get("content", ""))
            print("-" * 40)

    print("=" * 60)
    credential.close()


if __name__ == "__main__":
    asyncio.run(main())
