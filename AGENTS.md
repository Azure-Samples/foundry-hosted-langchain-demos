# Instructions for Coding Agents

The goal of this project is to be the teaching example for this livestream series:

https://developer.microsoft.com/en-us/reactor/series/S-1655/

## Host your agents on Foundry: LangChain / LangGraph

This repo mirrors the structure and teaching flow of
[`pamelafox/foundry-hosted-agentframework-demo`](https://github.com/pamelafox/foundry-hosted-agentframework-demo),
but uses the **LangChain/LangGraph** ecosystem instead of Microsoft Agent Framework.

Starting with a fully local agent (Stage 0 with Ollama), it progressively adds Foundry
capabilities: model deployments (Stage 1), Foundry IQ grounding via Azure AI Search
(Stage 2), Foundry Toolbox (Stage 3), and finally a fully hosted agent and workflow
service deployed with `azd`.

## Azure conventions

This repository follows Azure Developer CLI (`azd ai`) environment variable naming conventions.

Always use these names in code, scripts, and docs:

- `FOUNDRY_PROJECT_ENDPOINT`
- `AZURE_AI_MODEL_DEPLOYMENT_NAME`
- `AZURE_AI_SEARCH_SERVICE_ENDPOINT`
- `AZURE_AI_SEARCH_CONNECTION_NAME`

Do not introduce custom aliases unless there is a temporary backward-compatibility need.

Reference:

- https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/extensions/azure-ai-foundry-extension#manage-environment-variables

## Hosted agent environment variables

**Reserved prefixes:** The hosted agent platform reserves the `FOUNDRY_*` and `AGENT_*`
environment variable prefixes. Do not use these prefixes for custom variables in
`agent.yaml` — the deploy will fail with `invalid_payload`. If you need a custom variable
that references a Foundry concept, use a different prefix (e.g., `CUSTOM_FOUNDRY_AGENT_TOOLBOX_NAME`).

**Auto-injected variables:** The platform automatically injects these variables into the
container at runtime — do **not** set them in `agent.yaml`:

- `FOUNDRY_PROJECT_ENDPOINT` — the Foundry project endpoint
- `APPLICATIONINSIGHTS_CONNECTION_STRING` — the App Insights connection string for tracing

## Logging

Do NOT call `logging.basicConfig()` in hosted agent code (`main.py`) that uses
`from_langgraph` / `ResponsesHostServer`.

The agentserver SDK configures its own logging and OpenTelemetry tracing during startup.
Calling `logging.basicConfig()` on top of that adds duplicate `StreamHandler`s and every
log line appears twice with different formats.

Instead, get a named logger and set its level directly:

```python
logger = logging.getLogger("hr-agent")
logger.setLevel(logging.INFO)
```

The `stage*.py` files are local-only teaching scripts and **may** use
`logging.basicConfig()` since they do not run inside the agentserver.

## Dependency management

`pyproject.toml` and `uv.lock` exist in both the repo root and `workflows/`. Each
service has its own Docker build context and its own set of dependencies.

- The **root** `pyproject.toml` covers the hosted agent (`main.py`) and the staged demos.
- The **`workflows/`** `pyproject.toml` covers the hosted workflow service.

When you add or update a dependency that is shared between both services, update both
`pyproject.toml` files and regenerate both lock files:

```bash
uv lock
cd workflows && uv lock
```

## Testing hooks

To re-run azd hooks without a full deploy:

```bash
azd hooks run postprovision
azd hooks run postdeploy
```

## Linting

This repo uses [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
uv run ruff check .
```

## Filing bugs

This is where to search and file bugs for the technologies used in this repository:

* LangChain / LangGraph: github.com/langchain-ai/langchain
* LangChain MCP adapters: github.com/langchain-ai/langchain-mcp-adapters
* Agentserver LangGraph wrapper: github.com/azure/azure-sdk-for-python
* azd: github.com/Azure/azure-dev

## Relevant documentation

LangGraph documentation:
https://langchain-ai.github.io/langgraph/

LangChain Azure OpenAI integration:
https://python.langchain.com/docs/integrations/chat/azure_chat_openai/

Set up tracing in Foundry:
https://learn.microsoft.com/en-us/azure/foundry/observability/how-to/trace-agent-setup

Hosted agents overview:
https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/hosted-agents

Deploy a hosted agent (tutorial):
https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/deploy-hosted-agent?tabs=bash
