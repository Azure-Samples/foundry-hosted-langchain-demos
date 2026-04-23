#!/bin/sh
set -eu

echo "Running postdeploy hook: assigning Search role to hosted agent identity..."

if [ -z "${AZURE_AI_SEARCH_SERVICE_NAME:-}" ] || [ -z "${AZURE_SUBSCRIPTION_ID:-}" ] || [ -z "${AZURE_RESOURCE_GROUP:-}" ]; then
    echo "Search service or subscription info not set. Skipping role assignment."
    exit 0
fi

SEARCH_SCOPE="/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${AZURE_RESOURCE_GROUP}/providers/Microsoft.Search/searchServices/${AZURE_AI_SEARCH_SERVICE_NAME}"

SERVICES="hosted-langgraph-agent hosted-langgraph-workflow"
for SERVICE in $SERVICES; do
    AGENT_PRINCIPAL_ID=$(azd ai agent show "$SERVICE" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['instance_identity']['principal_id'])" 2>/dev/null || true)

    if [ -z "$AGENT_PRINCIPAL_ID" ]; then
        echo "Could not retrieve identity for $SERVICE. Skipping."
        continue
    fi

    echo "Assigning Search Index Data Contributor to $SERVICE ($AGENT_PRINCIPAL_ID)..."
    az role assignment create \
        --assignee-object-id "$AGENT_PRINCIPAL_ID" \
        --assignee-principal-type ServicePrincipal \
        --role "8ebe5a00-799e-43f5-93ac-243d3dce84a7" \
        --scope "$SEARCH_SCOPE" \
        --only-show-errors \
        --output none \
        2>/dev/null || echo "Role assignment may already exist for $SERVICE (this is OK)."
done

echo "Postdeploy hook complete."
