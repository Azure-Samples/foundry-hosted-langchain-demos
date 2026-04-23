"""
Workflow demo: Multi-agent workflow using LangGraph's StateGraph.

Three LLM nodes in a chain:
    writer → legal_reviewer → formatter

The writer creates a slogan, the legal reviewer checks it, and the formatter
styles it for terminal output. Each node only sees the output of the
previous node.
"""

import asyncio
import os

import httpx
from azure.ai.agentserver.langgraph import from_langgraph
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv(dotenv_path="../.env", override=True)

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT_NAME = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]


class _AzureTokenAuth(httpx.Auth):
    def __init__(self, provider):
        self._provider = provider

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self._provider()}"
        yield request


def _build_workflow():
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://ai.azure.com/.default"
    )
    model_http_client = httpx.Client(auth=_AzureTokenAuth(token_provider), timeout=120.0)

    llm = ChatOpenAI(
        base_url=f"{PROJECT_ENDPOINT.rstrip('/')}/openai/v1",
        api_key="placeholder",
        model=MODEL_DEPLOYMENT_NAME,
        use_responses_api=True,
        streaming=True,
        http_client=model_http_client,
    )

    async def writer(state: MessagesState) -> dict:
        user_input = state["messages"][-1].content
        response = await llm.ainvoke(
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

    async def legal_reviewer(state: MessagesState) -> dict:
        previous_output = state["messages"][-1].content
        response = await llm.ainvoke(
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

    async def formatter(state: MessagesState) -> dict:
        previous_output = state["messages"][-1].content
        response = await llm.ainvoke(
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


async def main() -> None:
    workflow = _build_workflow()
    await from_langgraph(workflow).run_async()


if __name__ == "__main__":
    asyncio.run(main())
