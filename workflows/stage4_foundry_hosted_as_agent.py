"""
Workflow demo: Multi-agent workflow using LangGraph's StateGraph.

Uses the Responses protocol via ResponsesAgentServerHost for hosting.
Three LLM nodes in a chain:
    writer → legal_reviewer → formatter

The writer creates a slogan, the legal reviewer checks it, and the formatter
styles it for terminal output. Each node only sees the output of the
previous node.

Conversation history is managed by the platform via
``previous_response_id`` and ``context.get_history()``.
"""

import asyncio
import logging
import os

import httpx
from azure.ai.agentserver.responses import (
    CreateResponse,
    ResponseContext,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
    TextResponse,
)
from azure.ai.agentserver.responses.models import (
    MessageContentInputTextContent,
    MessageContentOutputTextContent,
)
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv(dotenv_path="../.env", override=True)

logger = logging.getLogger("workflow-agent")
logger.setLevel(logging.INFO)

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT_NAME = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]

_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(_credential, "https://ai.azure.com/.default")


class _AzureTokenAuth(httpx.Auth):
    def __init__(self, provider):
        self._provider = provider

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self._provider()}"
        yield request


_http_client = httpx.Client(auth=_AzureTokenAuth(_token_provider), timeout=120.0)


def _build_workflow():
    llm = ChatOpenAI(
        base_url=f"{PROJECT_ENDPOINT.rstrip('/')}/openai/v1",
        api_key="placeholder",
        model=MODEL_DEPLOYMENT_NAME,
        use_responses_api=True,
        streaming=True,
        http_client=_http_client,
    )

    def writer(state: MessagesState) -> dict:
        user_input = state["messages"][-1].content
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an excellent slogan writer. "
                        "You create new slogans based on the given topic."
                    )
                ),
                HumanMessage(content=user_input),
            ]
        )
        return {"messages": [response]}

    def legal_reviewer(state: MessagesState) -> dict:
        previous_output = state["messages"][-1].content
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an excellent legal reviewer. "
                        "Make necessary corrections to the slogan so that it is legally compliant."
                    )
                ),
                HumanMessage(content=previous_output),
            ]
        )
        return {"messages": [response]}

    def formatter(state: MessagesState) -> dict:
        previous_output = state["messages"][-1].content
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an excellent content formatter. "
                        "You take the slogan and format it in Markdown with bold text and decorative elements. "
                        "Do not use ANSI escape codes or terminal color codes."
                    )
                ),
                HumanMessage(content=previous_output),
            ]
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("writer", writer)
    graph.add_node("legal_reviewer", legal_reviewer)
    graph.add_node("formatter", formatter)
    graph.add_edge(START, "writer")
    graph.add_edge("writer", "legal_reviewer")
    graph.add_edge("legal_reviewer", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()


WORKFLOW = _build_workflow()


# ── Responses protocol handler ──────────────────────────────────────


def _history_to_langchain_messages(history: list) -> list:
    """Convert responses-protocol history items to LangChain messages."""
    messages = []
    for item in history:
        if hasattr(item, "content") and item.content:
            for content in item.content:
                if isinstance(content, MessageContentOutputTextContent) and content.text:
                    messages.append(AIMessage(content=content.text))
                elif isinstance(content, MessageContentInputTextContent) and content.text:
                    messages.append(HumanMessage(content=content.text))
    return messages


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
            try:
                history = await context.get_history()
            except Exception:
                history = []
            current_input = await context.get_input_text() or "Hello!"

            lc_messages = _history_to_langchain_messages(history)
            lc_messages.append(HumanMessage(content=current_input))

            result = await WORKFLOW.ainvoke({"messages": lc_messages})

            raw = result["messages"][-1].content
            if isinstance(raw, list):
                yield "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in raw
                )
            else:
                yield raw or ""
        except Exception as exc:
            logger.exception("run_workflow failed")
            yield f"[ERROR] {type(exc).__name__}: {exc}"

    return TextResponse(context, request, text=run_workflow())


if __name__ == "__main__":
    app.run()
