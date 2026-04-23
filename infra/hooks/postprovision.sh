#!/bin/sh
set -eu

echo "Writing .env file from azd environment..."
bash infra/hooks/write_dot_env.sh

echo "Running postprovision hook for Foundry IQ (Azure AI Search)..."

uv run python infra/create-search-indexes.py

echo "Foundry IQ postprovision complete."

echo "Creating Foundry Toolbox..."

uv run python infra/create-toolbox.py

echo "Foundry Toolbox postprovision complete."
