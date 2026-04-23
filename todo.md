# TODO

## Pre-deploy checks

- [ ] Verify `uv sync` succeeds at repo root
- [ ] Verify `uv sync` succeeds in `workflows/`
- [ ] Run `uv run ruff check .` and fix any lint issues
- [ ] Confirm `main.py` starts locally with `azd ai agent run`
- [ ] Confirm `workflows/main.py` starts locally

## Deployment

- [ ] Run `azd auth login`
- [ ] Run `azd ai agent init`
- [ ] Run `azd up` — watch for build/deploy errors
- [ ] Run post-provision hooks: `azd hooks run postprovision`
- [ ] Run post-deploy hooks: `azd hooks run postdeploy`
- [ ] Populate search indexes: `uv run python infra/create-search-indexes.py`
- [ ] Create toolbox: `uv run python infra/create-toolbox.py`

## Post-deploy validation

- [ ] Test hosted agent in Foundry playground
- [ ] Test with `azd ai agent invoke` locally
- [ ] Test with `uv run python call_deployed_agent.py "test question"`
- [ ] Verify traces show up in Application Insights
- [ ] Test hosted workflow service is reachable

## Open questions

- [ ] Is `FOUNDRY_AGENT_TOOLBOX_FEATURES` in `main.py` a platform-provided var or should it be renamed to avoid the reserved `FOUNDRY_*` prefix?
- [ ] Should `AGENTS.md` be updated in the existing PR or committed separately?
- [ ] Are the `workflows/` pyproject.toml deps still current or do they need a version bump?
- [ ] Do the evaluation scripts (`scripts/quality_eval.py`, `scripts/red_team_scan.py`) work against the deployed agent?
