# Foundry-hosted LangChain demos

LangGraph-first demos for Microsoft Foundry that mirror the structure and teaching flow of [`pamelafox/foundry-hosted-agentframework-demo`](https://github.com/pamelafox/foundry-hosted-agentframework-demo), but use the LangChain/LangGraph ecosystem instead of Agent Framework.

## What this repo teaches

- **Stage 0:** a fully local tool-calling agent with Ollama
- **Stage 1:** the same agent backed by an Azure OpenAI / Foundry model deployment
- **Stage 2:** grounding with **Foundry IQ** through Azure AI Search
- **Stage 3:** toolbox tools through **Foundry Toolbox**
- **Hosted path:** a Foundry-hosted agent built with **LangGraph**, with toolbox tools and tracing
- **Workflows:** LangGraph workflows running as a separate Foundry-hosted service

This initial version is **LangGraph-first** while staying under the broader “LangChain” umbrella Pamela described.

## Repository shape

The repo intentionally follows Pamela's sample closely:

- `agents/stage0_local_agent.py`
- `agents/stage1_foundry_model.py`
- `agents/stage2_foundry_iq.py`
- `agents/stage2_foundry_iq_retrieve.py`
- `agents/stage2_foundry_iq_workaround.py`
- `agents/stage3_foundry_toolbox.py`
- `agents/stage4_foundry_hosted.py` — hosted agent entry point
- `agents/agent.yaml` — hosted agent configuration
- `agents/call_deployed_agent.py` — invoke a deployed agent from the CLI
- `azure.yaml` — azd service definitions (agent + workflow)
- `infra/`
- `data/index-data/`
- `scripts/`
- `workflows/` — LangGraph workflow demos (separate hosted service)

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Azure Developer CLI (`azd`)](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd) with the AI agent extension
- An Azure subscription

Optional for Stage 0:

- [Ollama](https://ollama.com/download)

## Quick start

### Install dependencies

```bash
uv sync
```

### Prepare environment

```bash
cp .env.sample .env
```

Fill in the variables you need for the stage you want to run.

### Run the staged demos

```bash
uv run python agents/stage0_local_agent.py
uv run python agents/stage1_foundry_model.py
uv run python agents/stage2_foundry_iq.py
```

If your KB MCP endpoint has compatibility issues, use one of the fallbacks:

```bash
uv run python agents/stage2_foundry_iq_workaround.py
uv run python agents/stage2_foundry_iq_retrieve.py
```

After creating the toolbox with the deploy steps below, you can also run:

```bash
uv run python agents/stage3_foundry_toolbox.py
```

## Deploy the hosted agent

```bash
azd auth login
azd ai agent init
azd up
```

The `azd up` command provisions infrastructure, creates search indexes and knowledge base,
creates the toolbox, and deploys both the agent and workflow services.

Run locally:

```bash
azd ai agent run
```

Invoke from another terminal:

```bash
azd ai agent invoke --local "What PerksPlus benefits are there, and when do I need to enroll by?"
```

Or call a deployed agent via the SDK:

```bash
uv run python agents/call_deployed_agent.py "What PerksPlus benefits are there?"
```

## Workflows

The `workflows/` directory is deployed as a separate Foundry-hosted service and demonstrates LangGraph workflows:

- `workflows/simple_workflow.py` — pure data-transformation pipeline (no LLM)
- `workflows/stage1_foundry_model.py` — workflow backed by a Foundry model
- `workflows/workflow_as_agent.py` — writer → formatter pipeline with streaming
- `workflows/stage4_foundry_hosted_as_agent.py` — hosted workflow entry point

## Evaluation scripts

The `scripts/` folder mirrors Pamela's repo and includes:

- `scripts/quality_eval.py`
- `scripts/red_team_scan.py`
- `scripts/scheduled_eval.py`
- `scripts/scheduled_red_team.py`

Run them with:

```bash
uv run scripts/quality_eval.py
uv run scripts/red_team_scan.py
```

## Observability

The hosted agent is designed to run in Foundry so you can:

- test it in the **Foundry playground**
- inspect **tool calls and traces**
- use **Application Insights** for deeper diagnostics

For local debugging, set `APPLICATIONINSIGHTS_CONNECTION_STRING` in `.env`.

## Related repos

- Structural source: [`pamelafox/foundry-hosted-agentframework-demo`](https://github.com/pamelafox/foundry-hosted-agentframework-demo)
- Hosted LangGraph reference patterns: `microsoft/hosted-agents-vnext-private-preview`
