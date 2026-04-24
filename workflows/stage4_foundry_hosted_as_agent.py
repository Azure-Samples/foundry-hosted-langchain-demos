"""
Workflow demo: Multi-step workflow using LangGraph's Functional API.

Three LLM tasks in a chain:
    writer → legal_reviewer → formatter

The writer creates a slogan, the legal reviewer checks it, and the formatter
styles it for terminal output. Each task only sees the output of the
previous task.

This module uses AzureAIResponsesAgentHost from a vendored copy of
https://github.com/langchain-ai/langchain-azure/pull/501 which provides
first-class LangGraph hosting support for Azure AI Foundry.
"""

import logging
import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task

from _vendor.langchain_azure_ai_runtime import (
    AzureAIResponsesAgentHost,
    ResponsesInputContext,
    ResponsesInputRequest,
)

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
)


@task
def writer(user_input: str) -> str:
    """Create a slogan based on the user's input."""
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an excellent slogan writer. "
                    "You create new slogans based on the given topic."
                ),
            },
            {"role": "user", "content": user_input},
        ]
    )
    return response.content


@task
def legal_reviewer(text: str) -> str:
    """Review and correct the slogan for legal compliance."""
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an excellent legal reviewer. "
                    "Make necessary corrections to the slogan so that it is legally compliant."
                ),
            },
            {"role": "user", "content": text},
        ]
    )
    return response.content


@task
def formatter(text: str) -> str:
    """Format the slogan with Markdown for display."""
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an excellent content formatter. "
                    "You take the slogan and format it in Markdown with bold text "
                    "and decorative elements. Do not use ANSI escape codes or terminal color codes."
                ),
            },
            {"role": "user", "content": text},
        ]
    )
    return response.content


@entrypoint()
def workflow(user_input: str) -> str:
    """Chain: Writer → Legal Reviewer → Formatter."""
    draft = writer(user_input).result()
    reviewed = legal_reviewer(draft).result()
    return formatter(reviewed).result()


# ── Custom parsers for str-based workflow graph ─────────────────────


async def workflow_input_parser(
    request: ResponsesInputRequest,
    context: ResponsesInputContext,
) -> str:
    """Extract user text from the Foundry request as a plain string."""
    return await context.get_input_text() or "Hello!"


def workflow_output_parser(item: object) -> str:
    """Extract text from a workflow stream item (values mode yields str)."""
    if isinstance(item, str):
        return item
    return ""


# ── Hosted agent entrypoint ─────────────────────────────────────────

host = AzureAIResponsesAgentHost(
    graph=workflow,
    stream_mode="values",
    responses_history_count=20,
    input_parser=workflow_input_parser,
    output_parser=workflow_output_parser,
)

if __name__ == "__main__":
    host.run()
