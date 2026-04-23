"""
Hosted HR helper built with LangGraph and Microsoft Foundry.

Uses the Responses protocol via ResponsesAgentServerHost for hosting.
The LangGraph agent has two nodes: chatbot (calls LLM with tools) and
tools (executes tool calls). Conversation history is managed by the
platform via ``previous_response_id`` and ``context.get_history()``.

Run locally with:
    azd ai agent run
"""

import asyncio
import json
import logging
import os
from datetime import date
from typing import Annotated

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
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import Field
from typing_extensions import TypedDict

load_dotenv(dotenv_path=".env", override=True)

logger = logging.getLogger("hr-agent")
logger.setLevel(logging.INFO)

if not os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    logger.warning(
        "APPLICATIONINSIGHTS_CONNECTION_STRING not set — traces will not be sent to "
        "Application Insights. Set it for local telemetry; hosted containers inject it automatically."
    )

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT_NAME = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]
SEARCH_SERVICE_ENDPOINT = os.environ["AZURE_AI_SEARCH_SERVICE_ENDPOINT"]
KNOWLEDGE_BASE_NAME = os.environ["AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME"]
TOOLBOX_NAME = os.environ.get("CUSTOM_FOUNDRY_AGENT_TOOLBOX_NAME", "hr-agent-tools")
TOOLBOX_FEATURES = os.getenv("FOUNDRY_AGENT_TOOLBOX_FEATURES", "Toolboxes=V1Preview")

_credential = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(_credential, "https://ai.azure.com/.default")


class _AzureTokenAuth(httpx.Auth):
    def __init__(self, provider):
        self._provider = provider

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self._provider()}"
        yield request


_http_client = httpx.Client(auth=_AzureTokenAuth(_token_provider), timeout=120.0)


class KnowledgeBaseMCPTool:
    """Manual MCP wrapper for Azure AI Search knowledge-base retrieval."""

    def __init__(self, http_client: httpx.Client, mcp_url: str) -> None:
        self._http_client = http_client
        self._mcp_url = mcp_url
        self._initialized = False
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._http_client.post(
            self._mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"sampling": {}},
                    "clientInfo": {"name": "hr-agent", "version": "0.1.0"},
                },
            },
            headers=self._headers,
        ).raise_for_status()
        self._http_client.post(
            self._mcp_url,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=self._headers,
        ).raise_for_status()
        self._initialized = True

    def retrieve(self, queries: list[str]) -> str:
        self._ensure_initialized()
        response = self._http_client.post(
            self._mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "knowledge_base_retrieve",
                    "arguments": {"queries": queries},
                },
            },
            headers=self._headers,
        )
        response.raise_for_status()
        for line in response.text.splitlines():
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:].strip())
            result = data.get("result", {})
            content = result.get("content", [])
            snippets: list[str] = []
            for item in content:
                if item.get("type") == "resource" and "resource" in item:
                    snippets.append(item["resource"].get("text", ""))
                elif item.get("type") == "text":
                    snippets.append(item.get("text", ""))
            if snippets:
                return "\n\n---\n\n".join(snippets)
        return "No results found."


def _sanitize_tools(tools: list) -> list:
    for tool_obj in tools:
        tool_obj.handle_tool_error = True
        schema = tool_obj.args_schema if isinstance(tool_obj.args_schema, dict) else None
        if schema is None:
            continue
        if schema.get("type") == "object" and "properties" not in schema:
            schema["properties"] = {}
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if required and not props:
            for field_name in required:
                props[field_name] = {"type": "string"}
            schema["properties"] = props
    return tools


@tool
def get_current_date() -> str:
    """Return the current date in ISO format."""
    return date.today().isoformat()


@tool
def get_enrollment_deadline_info() -> dict:
    """Return enrollment timeline details for health insurance plans."""
    return {
        "benefits_enrollment_opens": "2026-11-11",
        "benefits_enrollment_closes": "2026-11-30",
    }


SYSTEM_PROMPT = """You are an internal HR helper focused on employee benefits and company information.

Use the knowledge-base tool first for questions about Zava policies, benefits, plans,
deadlines, and internal company information.

Use toolbox tools such as web search when the knowledge base does not have the answer
or when the user asks for current external information.

Use get_enrollment_deadline_info and get_current_date when the question involves
benefits timing.

If the tools do not provide enough information, say so clearly and do not invent facts.
"""


# ── LangGraph definition ────────────────────────────────────────────


class State(TypedDict):
    messages: Annotated[list, add_messages]


_graph = None
_toolbox_client = None
_graph_lock = asyncio.Lock()


async def _build_graph():
    global _toolbox_client

    search_token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://search.azure.com/.default"
    )
    kb_http_client = httpx.Client(
        auth=_AzureTokenAuth(search_token_provider),
        timeout=httpx.Timeout(30.0, read=300.0),
    )
    kb_tool = KnowledgeBaseMCPTool(
        kb_http_client,
        (
            f"{SEARCH_SERVICE_ENDPOINT.rstrip('/')}"
            f"/knowledgebases/{KNOWLEDGE_BASE_NAME}/mcp?api-version=2025-11-01-Preview"
        ),
    )

    @tool
    def knowledge_base_retrieve(
        queries: Annotated[
            list[str],
            Field(
                description=(
                    "1 to 4 concise search queries (max ~12 words each). "
                    "Use alternate wording as separate entries."
                ),
                min_length=1,
                max_length=4,
            ),
        ],
    ) -> str:
        """Search the Zava company knowledge base for HR policies and benefits."""
        logger.info("KB retrieve: %s", queries)
        return kb_tool.retrieve(queries)

    # The hosted platform auto-injects FOUNDRY_AGENT_TOOLBOX_ENDPOINT; fall back to
    # constructing it manually for local development.
    toolbox_endpoint = os.environ.get(
        "FOUNDRY_AGENT_TOOLBOX_ENDPOINT",
        f"{PROJECT_ENDPOINT.rstrip('/')}/toolboxes/{TOOLBOX_NAME}/mcp?api-version=v1",
    )
    extra_headers = {"Foundry-Features": TOOLBOX_FEATURES} if TOOLBOX_FEATURES else {}
    _toolbox_client = MultiServerMCPClient(
        {
            "toolbox": {
                "url": toolbox_endpoint,
                "transport": "streamable_http",
                "headers": extra_headers,
                "auth": _AzureTokenAuth(_token_provider),
            }
        }
    )

    toolbox_tools = []
    try:
        toolbox_tools = _sanitize_tools(await _toolbox_client.get_tools())
        logger.info("Loaded %d toolbox tools", len(toolbox_tools))
    except Exception as exc:
        logger.warning("Toolbox startup skipped: %s", exc)

    all_tools = [
        knowledge_base_retrieve,
        get_enrollment_deadline_info,
        get_current_date,
        *toolbox_tools,
    ]

    llm = ChatOpenAI(
        base_url=f"{PROJECT_ENDPOINT.rstrip('/')}/openai/v1",
        api_key="placeholder",
        model=MODEL_DEPLOYMENT_NAME,
        use_responses_api=True,
        streaming=True,
        http_client=_http_client,
    )
    llm_with_tools = llm.bind_tools(all_tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def route_tools(state: State):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", ToolNode(tools=all_tools))
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "chatbot")
    return graph.compile()


async def _get_graph():
    global _graph
    if _graph is not None:
        return _graph
    async with _graph_lock:
        if _graph is None:
            _graph = await _build_graph()
    return _graph


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
    """Run the LangGraph agent and stream the response."""

    async def run_graph():
        try:
            graph = await _get_graph()
            try:
                history = await context.get_history()
            except Exception:
                history = []
            current_input = await context.get_input_text() or "Hello!"

            lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
            lc_messages.extend(_history_to_langchain_messages(history))
            lc_messages.append(HumanMessage(content=current_input))

            result = await graph.ainvoke({"messages": lc_messages})

            # With use_responses_api, content may be a list of content blocks.
            raw = result["messages"][-1].content
            if isinstance(raw, list):
                yield "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in raw
                )
            else:
                yield raw or ""
        except Exception as exc:
            logger.exception("run_graph failed")
            yield f"[ERROR] {type(exc).__name__}: {exc}"

    return TextResponse(context, request, text=run_graph())


if __name__ == "__main__":
    app.run()
