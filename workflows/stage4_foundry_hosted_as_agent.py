"""
Workflow demo: Multi-step workflow using LangGraph's Functional API.

Three LLM tasks in a chain:
    writer → legal_reviewer → formatter

The writer creates a slogan, the legal reviewer checks it, and the formatter
styles it for terminal output. Each task only sees the output of the
previous task.
"""

import asyncio
import logging
import os

from azure.ai.agentserver.responses import (
    CreateResponse,
    ResponseContext,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
    TextResponse,
)
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task

load_dotenv(dotenv_path="../.env", override=True)

logger = logging.getLogger("workflow-agent")
logger.setLevel(logging.INFO)

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


# ── Responses protocol handler ──────────────────────────────────────

app = ResponsesAgentServerHost(
    options=ResponsesServerOptions(default_fetch_history_count=20)
)


@app.response_handler
async def handle_create(
    request: CreateResponse,
    context: ResponseContext,
    cancellation_signal: asyncio.Event,
):
    """Run the workflow and stream the response."""

    async def run_workflow():
        try:
            current_input = await context.get_input_text() or "Hello!"
            result = await workflow.ainvoke(current_input)

            if isinstance(result, list):
                yield "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in result
                )
            else:
                yield result or ""
        except Exception as exc:
            logger.exception("run_workflow failed")
            yield f"[ERROR] {type(exc).__name__}: {exc}"

    return TextResponse(context, request, text=run_workflow())


if __name__ == "__main__":
    app.run()
