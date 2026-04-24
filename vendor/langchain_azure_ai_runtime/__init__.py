# ---------------------------------------------------------------------------
# Vendored from: https://github.com/langchain-ai/langchain-azure/pull/501
# Original author: santiagxf (https://github.com/santiagxf)
# Reason: PR adds first-class LangGraph hosted-agent support for Azure AI
#         Foundry but is unlikely to be merged due to lack of maintenance.
#         Vendored here so our demo can use AzureAIResponsesAgentHost.
# ---------------------------------------------------------------------------
# Copyright (c) Microsoft. All rights reserved.

"""Host a compiled LangGraph graph inside Azure AI Foundry's Agent Service.

Agent Service can host agents created in LangChain/LangGraph and serve them
with the same platform guarantees Foundry provides.

You can serve and hook your agent using OpenAI Responses API or a custom
API of your choice (called Invocations API). When using OpenAI Responses
API, Microsoft Foundry handles state automatically and securely stores it
within the service. Invocations API is a more generic approach that allow
you to use input and output schemas of your choice.

To run your agent in Foundry, use either `AzureAIInvokeAgentHost` and
`AzureAIResponsesAgentHost` depending on the API you want to use.

Quick start::

    ```python
    from langgraph.graph import StateGraph, MessagesState, START, END
    from vendor.langchain_azure_ai_runtime import (
        AzureAIResponsesAgentHost,
    )

    builder = StateGraph(MessagesState)
    builder.add_node("agent", my_agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    host = AzureAIResponsesAgentHost(
        graph=graph,
    )

    if __name__ == "__main__":
        host.run()
    ```

If you have a `langgraph.json` file, you can load the graph with:

    ```python
    from vendor.langchain_azure_ai_runtime import (
        AzureAIResponsesAgentHost,
    )

    host = AzureAIResponsesAgentHost.from_config()

    if __name__ == "__main__":
        host.run()
    ```


Error handling overview:

``AzureAIInvokeAgentHost``
    Uses an HTTP request/response model. Handled parser failures are returned
    as JSON error payloads, while graph/runtime failures outside those parser
    hooks are delegated to the underlying invocation server.

``AzureAIResponsesAgentHost``
    Uses a streaming Responses API model. Custom parser failures are surfaced
    as ``response.failed`` lifecycle events on the stream, while default
    request validation and non-parser runtime failures continue through the
    underlying Responses pipeline.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vendor.langchain_azure_ai_runtime._invoke_host import (
        AzureAIInvokeAgentHost,
        GraphInvocationInput,
        InvokeInputParser,
        InvokeInputRequest,
        InvokeOutputParser,
        InvokeOutputResponse,
    )
    from vendor.langchain_azure_ai_runtime._responses_host import (
        AzureAIResponsesAgentHost,
        ResponsesInputContext,
        ResponsesInputParser,
        ResponsesInputRequest,
        ResponsesOutputItem,
        ResponsesOutputParser,
    )

__all__ = [
    "AzureAIResponsesAgentHost",
    "AzureAIInvokeAgentHost",
    "GraphInvocationInput",
    "InvokeInputRequest",
    "InvokeInputParser",
    "InvokeOutputParser",
    "InvokeOutputResponse",
    "ResponsesInputRequest",
    "ResponsesInputContext",
    "ResponsesInputParser",
    "ResponsesOutputItem",
    "ResponsesOutputParser",
]

_module_lookup = {
    "AzureAIInvokeAgentHost": "vendor.langchain_azure_ai_runtime._invoke_host",
    "GraphInvocationInput": "vendor.langchain_azure_ai_runtime._invoke_host",
    "AzureAIResponsesAgentHost": "vendor.langchain_azure_ai_runtime._responses_host",
    "InvokeInputRequest": "vendor.langchain_azure_ai_runtime._invoke_host",
    "InvokeInputParser": "vendor.langchain_azure_ai_runtime._invoke_host",
    "InvokeOutputParser": "vendor.langchain_azure_ai_runtime._invoke_host",
    "InvokeOutputResponse": "vendor.langchain_azure_ai_runtime._invoke_host",
    "ResponsesInputRequest": "vendor.langchain_azure_ai_runtime._responses_host",
    "ResponsesInputContext": "vendor.langchain_azure_ai_runtime._responses_host",
    "ResponsesInputParser": "vendor.langchain_azure_ai_runtime._responses_host",
    "ResponsesOutputItem": "vendor.langchain_azure_ai_runtime._responses_host",
    "ResponsesOutputParser": "vendor.langchain_azure_ai_runtime._responses_host",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
