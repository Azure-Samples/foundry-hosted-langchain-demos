$ErrorActionPreference = "Stop"

Write-Host "Writing .env file from azd environment..."
& ./infra/hooks/write_dot_env.ps1

Write-Host "Running postprovision hook for Foundry IQ (Azure AI Search)..."

uv run python infra/create-search-indexes.py

Write-Host "Foundry IQ postprovision complete."

Write-Host "Creating Foundry Toolbox..."

uv run python infra/create-toolbox.py

Write-Host "Foundry Toolbox postprovision complete."
