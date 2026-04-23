"""
Workflow Stage 2: Writer → Formatter workflow using a Foundry-hosted model.

Two LLM nodes (agents) in a chain:
    writer → formatter

The writer drafts a short article, and the formatter styles it with
Markdown and emojis. Each node only sees the output of the previous node.

Prerequisites:
    - An Azure OpenAI / Foundry model deployment
    - `az login` (uses DefaultAzureCredential)
    - .env with:
        AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
        AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-5.2

Run:
    python workflows/stage2_agent_nodes.py
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
    """Run a writer → formatter workflow against a Foundry-hosted model."""
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )

    llm = ChatOpenAI(
        base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/v1/",
        api_key=token_provider,
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
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

    prompt = 'Write a 2-sentence LinkedIn post: "Why your AI pilot looks good but fails in production."'
    print(f"\nPrompt: {prompt}\n")

    result = await workflow.ainvoke({"content": prompt})
    print("Output:")
    print(result["content"])

    credential.close()


if __name__ == "__main__":
    asyncio.run(main())
