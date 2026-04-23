# Internal HR Benefits Agent

A sample AI agent built with [LangChain/LangGraph](https://langchain-ai.github.io/langgraph/) that helps employees with HR benefits questions. This project is designed as an `azd` starter template for deploying hosted AI agents to [Microsoft Foundry](https://learn.microsoft.com/azure/foundry/).

This repo mirrors the structure and teaching flow of [`pamelafox/foundry-hosted-agentframework-demo`](https://github.com/pamelafox/foundry-hosted-agentframework-demo), but uses the **LangChain/LangGraph** ecosystem instead of Microsoft Agent Framework.

## What it does

The agent uses company HR documents (via Azure AI Search) and tool-calling to:

- Answer questions about employee benefits (health insurance, dental, vision, 401k, etc.)
- Look up enrollment deadlines and dates
- Search the web for current information when the knowledge base doesn't have the answer
- Run code via Code Interpreter for data analysis tasks

## Architecture

The agent connects to a **Foundry Toolbox** MCP endpoint that provides knowledge-base retrieval, web search, and code interpreter as tools. The LangGraph agent decides when to call each tool based on the user's question.

The staged demos show the progression from local to hosted:

| Stage | Script | What it adds |
|-------|--------|-------------|
| 0 | `agents/stage0_local_model.py` | Fully local agent with Ollama |
| 1 | `agents/stage1_foundry_model.py` | Azure OpenAI / Foundry model deployment |
| 2 | `agents/stage2_foundry_iq.py` | Foundry IQ grounding via Azure AI Search |
| 3 | `agents/stage3_foundry_toolbox.py` | Foundry Toolbox (KB + web search + code interpreter) |
| 4 | `agents/stage4_foundry_hosted.py` | Hosted agent with Responses protocol |

## Prerequisites

- [Python 3.12+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Azure Developer CLI (azd) 1.23.7+](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd)
- An [Azure subscription](https://azure.microsoft.com/free/)

Optional for Stage 0:

- [Ollama](https://ollama.com/download)

## Quick start

### Deploy to Azure

```bash
azd auth login
azd ai agent init
azd up
```

During `azd ai agent init`, you'll be prompted to choose a model. Select `gpt-5.2` or another supported model.

> **Region:** The template restricts deployment to regions that support all features (Responses API, evaluations, red teaming): `eastus2`, `francecentral`, `northcentralus`, `swedencentral`.

### Set up the knowledge base

After provisioning, create the search indexes and knowledge base:

```bash
./write_dot_env.sh  # or .\write_dot_env.ps1 on Windows
uv run python infra/create-search-indexes.py \
    --endpoint "$AZURE_AI_SEARCH_SERVICE_ENDPOINT" \
    --openai-endpoint "$AZURE_OPENAI_ENDPOINT" \
    --openai-model-deployment "$AZURE_AI_MODEL_DEPLOYMENT_NAME"
```

This creates:
- `hrdocs` and `healthdocs` search indexes with sample data
- A single knowledge base (`zava-company-kb`) with both indexes as knowledge sources

### Run locally

1. Sync your `.env` from the azd environment:

    ```bash
    ./write_dot_env.sh
    ```

2. Start the local hosted-agent server:

    ```bash
    azd ai agent run
    ```

3. Invoke the agent from another terminal:

    ```bash
    azd ai agent invoke --local "What benefits are there, and when do I need to enroll by?"
    ```

### Deploy the agent

```bash
azd deploy
```

### Run the staged demos

The staged scripts can be run independently for teaching purposes:

```bash
uv run python agents/stage0_local_model.py
uv run python agents/stage1_foundry_model.py
uv run python agents/stage2_foundry_iq.py
uv run python agents/stage3_foundry_toolbox.py
```

Or call a deployed agent via the SDK:

```bash
uv run python agents/call_foundry_hosted.py "What PerksPlus benefits are there?"
```

## Workflows

The `workflows/` directory is deployed as a separate Foundry-hosted service and demonstrates LangGraph workflows:

| Stage | Script | What it demonstrates |
|-------|--------|---------------------|
| 1 | `workflows/stage1_simple_nodes.py` | Pure data-transformation pipeline (no LLM) |
| 2 | `workflows/stage2_agent_nodes.py` | Workflow backed by a Foundry model |
| 3 | `workflows/stage3_as_agent.py` | Writer → formatter pipeline with streaming |
| 4 | `workflows/stage4_foundry_hosted_as_agent.py` | Hosted workflow entry point |

## Evaluation scripts

Scripts for quality evaluation, red teaming, and scheduled runs are in `scripts/`:

| Script | Description |
|--------|-------------|
| `scripts/quality_eval.py` | Run quality evaluation (task adherence, groundedness, relevance) |
| `scripts/red_team_scan.py` | Run a one-time red team scan with attack strategies |
| `scripts/scheduled_eval.py` | Set up daily quality evaluation schedule |
| `scripts/scheduled_red_team.py` | Set up daily red team schedule |

```bash
uv run scripts/quality_eval.py
uv run scripts/red_team_scan.py
```

> **Note:** Red teaming requires a supported region (East US 2, Sweden Central, etc.). See [evaluation region support](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-regions-limits-virtual-network).

## Debug with `azd`

After deploying, use these commands to inspect and troubleshoot your hosted agent:

```bash
# View container status, health, and error details
azd ai agent show

# Fetch recent logs
azd ai agent monitor

# Stream logs in real time
azd ai agent monitor -f
```

## Observability

The hosted agent server exports its own HTTP-layer traces (request/response timing) to Application Insights automatically when `APPLICATIONINSIGHTS_CONNECTION_STRING` is set.

To capture sensitive data in traces (tool call arguments, prompts, responses), set `enable_content_recording=True` in the `enable_auto_tracing()` call. This is useful for debugging but should be disabled in production.

To query traces in Application Insights:

```kql
dependencies
| where timestamp > ago(1h)
| where customDimensions has "gen_ai.operation.name"
| extend opName = tostring(customDimensions["gen_ai.operation.name"])
| extend toolName = tostring(customDimensions["gen_ai.tool.name"])
| extend toolArgs = tostring(customDimensions["gen_ai.tool.call.arguments"])
| project timestamp, name, opName, toolName, toolArgs
| order by timestamp desc
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FOUNDRY_PROJECT_ENDPOINT` | Yes | Foundry project endpoint |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | Yes | Model deployment name (e.g., `gpt-5.2`) |
| `AZURE_OPENAI_ENDPOINT` | Stages only | Azure OpenAI endpoint for local stage scripts |
| `AZURE_AI_SEARCH_SERVICE_ENDPOINT` | Stage 2 only | Azure AI Search endpoint |
| `AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME` | Stage 2 only | Knowledge base name (default: `zava-company-kb`) |
| `CUSTOM_FOUNDRY_AGENT_TOOLBOX_NAME` | No | Toolbox name (default: `hr-agent-tools`) |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | No | App Insights connection string for tracing |
