"""Create (or update) the Foundry Toolbox with web search and code interpreter tools.

Usage:
    uv run python infra/create-toolbox.py

Requires environment variables:
    FOUNDRY_PROJECT_ENDPOINT  — Foundry project endpoint URL
    CUSTOM_FOUNDRY_AGENT_TOOLBOX_NAME — Toolbox name (default: hr-agent-tools)
"""

import os

import httpx
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

_API_VERSION = "v1"
_SCOPE = "https://ai.azure.com/.default"
_FEATURE_HEADER = "Toolboxes=V1Preview"


def _headers(credential: DefaultAzureCredential) -> dict:
    token = credential.get_token(_SCOPE).token
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Foundry-Features": _FEATURE_HEADER,
    }


def create_or_update_toolbox(endpoint: str, toolbox_name: str) -> None:
    """Create a new version of the toolbox with web search and code interpreter tools."""
    credential = DefaultAzureCredential()
    base_url = f"{endpoint.rstrip('/')}/toolboxes/{toolbox_name}"

    tools = [
        {"type": "web_search", "name": "web_search"},
        {"type": "code_interpreter", "name": "code_interpreter"},
    ]

    # 1. Create a new version
    print(f"Creating toolbox '{toolbox_name}' at {endpoint} ...")
    resp = httpx.post(
        f"{base_url}/versions",
        params={"api-version": _API_VERSION},
        headers=_headers(credential),
        json={
            "tools": tools,
            "description": "Web search, code interpreter, and knowledge base tools for the HR agent.",
        },
        timeout=60,
    )
    if not resp.is_success:
        print(f"Create version failed ({resp.status_code}): {resp.text}")
        resp.raise_for_status()
    version = resp.json().get("version")
    print(f"Toolbox '{toolbox_name}' version {version} created.")

    # 2. Promote this version to default
    resp = httpx.patch(
        base_url,
        params={"api-version": _API_VERSION},
        headers=_headers(credential),
        json={"default_version": version},
        timeout=60,
    )
    resp.raise_for_status()
    print(f"Toolbox '{toolbox_name}' default version set to {version}.")


if __name__ == "__main__":
    endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
    toolbox_name = os.environ.get("CUSTOM_FOUNDRY_AGENT_TOOLBOX_NAME", "hr-agent-tools")

    create_or_update_toolbox(endpoint, toolbox_name)