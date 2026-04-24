"""
Workflow Stage 2: Writer → Formatter workflow using a Foundry-hosted model.

Two LLM tasks in a chain:
    writer → formatter

The writer drafts a short article, and the formatter styles it with
Markdown and emojis. Each task only sees the output of the previous task.

Prerequisites:
    - An Azure OpenAI / Foundry model deployment
    - `az login` (uses DefaultAzureCredential)
    - .env with:
        AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
        AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-5.2

Run:
    uv run python workflows/stage2_agent_nodes.py
"""

import asyncio
import os

from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task

load_dotenv(override=True)


async def main():
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )

    llm = ChatOpenAI(
        base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/v1/",
        api_key=token_provider,
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        use_responses_api=True,
    )

    @task
    async def writer(topic: str) -> str:
        """Draft a short article based on the given topic."""
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
                {"role": "user", "content": topic},
            ]
        )
        return response.content

    @task
    async def formatter(text: str) -> str:
        """Format text with Markdown and emojis."""
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
                {"role": "user", "content": text},
            ]
        )
        return response.content

    @entrypoint()
    async def workflow(topic: str) -> str:
        """Chain: Writer → Formatter."""
        draft = await writer(topic)
        return await formatter(draft)

    prompt = 'Write a 2-sentence LinkedIn post: "Why your AI pilot looks good but fails in production."'
    print(f"\nPrompt: {prompt}\n")
    result = await workflow.ainvoke(prompt)
    print("Output:")
    print(result)

    await credential.close()


if __name__ == "__main__":
    asyncio.run(main())
