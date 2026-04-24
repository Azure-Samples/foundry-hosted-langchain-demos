"""
Workflow as Agent: Writer → Formatter pipeline exposed as an agent.

What changes from Stage 2:
    - Switch from plain string I/O to MessagesState — the workflow now
      accepts user messages and responds with AI messages, just like an agent.
    - This is the same interface the hosted version uses in
      stage4_foundry_hosted_as_agent.py.

Prerequisites (same as Stage 2):
    - An Azure OpenAI / Foundry model deployment
    - `az login` (uses DefaultAzureCredential)
    - .env with:
        AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
        AZURE_AI_MODEL_DEPLOYMENT_NAME=gpt-5.2

Run:
    uv run python workflows/stage3_as_agent.py
"""

import asyncio
import os

from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task
from langgraph.graph import MessagesState

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
    async def writer(state: MessagesState) -> MessagesState:
        """Draft a short article based on the given topic."""
        user_text = state["messages"][-1].content
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
                {"role": "user", "content": user_text},
            ]
        )
        return {"messages": [AIMessage(content=response.content)]}

    @task
    async def formatter(state: MessagesState) -> MessagesState:
        """Format text with Markdown and emojis."""
        previous_output = state["messages"][-1].content
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
                {"role": "user", "content": previous_output},
            ]
        )
        return {"messages": [AIMessage(content=response.content)]}

    @entrypoint()
    async def workflow(state: MessagesState) -> MessagesState:
        """Chain: Writer → Formatter."""
        result = await writer(state)
        return await formatter(result)

    prompt = "Write a short post about why open-source AI frameworks matter."
    print(f"Prompt: {prompt}\n")
    result = await workflow.ainvoke({"messages": [HumanMessage(content=prompt)]})
    print("Output:")
    print(result["messages"][-1].content)

    await credential.close()


if __name__ == "__main__":
    asyncio.run(main())
