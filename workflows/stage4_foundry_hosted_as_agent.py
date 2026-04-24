"""
Workflow demo: Writer → Formatter workflow hosted on Azure AI Foundry.

Two LLM tasks in a chain:
    writer → formatter

The writer drafts a short article, and the formatter styles it with Markdown and emojis.
Each task only sees the output of the previous task.

This module uses AzureAIResponsesAgentHost from a vendored copy of
https://github.com/langchain-ai/langchain-azure/pull/501 which provides
first-class LangGraph hosting support for Azure AI Foundry.

Run using:
    azd ai agent run
"""

import logging
import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task
from langgraph.graph import MessagesState

from _vendor.langchain_azure_ai_runtime import AzureAIResponsesAgentHost

load_dotenv(dotenv_path="../.env", override=True)

logger = logging.getLogger("workflow-agent")
logger.setLevel(logging.INFO)

# Emit LangChain/LangGraph spans to Application Insights with gen_ai.agent.id
# so the Foundry portal Agent Monitor can identify this agent's traces.
enable_auto_tracing(
    auto_configure_azure_monitor=True,
    enable_content_recording=False,
    trace_all_langgraph_nodes=True,
    agent_id="slogan-workflow",
)

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT_NAME = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]

_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(_credential, "https://ai.azure.com/.default")

llm = ChatOpenAI(
    base_url=f"{PROJECT_ENDPOINT.rstrip('/')}/openai/v1",
    api_key=_token_provider,
    model=MODEL_DEPLOYMENT_NAME,
    use_responses_api=True,
    streaming=True,
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


# ── Hosted agent entrypoint ─────────────────────────────────────────

host = AzureAIResponsesAgentHost(
    graph=workflow,
    stream_mode="messages",
    responses_history_count=20,
)

if __name__ == "__main__":
    host.run()
