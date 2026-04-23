Write-Host "Running postdeploy hook: assigning Search role to hosted agent identity..."

if (-not $env:AZURE_AI_SEARCH_SERVICE_NAME -or -not $env:AZURE_SUBSCRIPTION_ID -or -not $env:AZURE_RESOURCE_GROUP) {
    Write-Host "Search service or subscription info not set. Skipping role assignment."
    exit 0
}

$searchScope = "/subscriptions/$env:AZURE_SUBSCRIPTION_ID/resourceGroups/$env:AZURE_RESOURCE_GROUP/providers/Microsoft.Search/searchServices/$env:AZURE_AI_SEARCH_SERVICE_NAME"

$services = @("hosted-langgraph-agent", "hosted-langgraph-workflow")
foreach ($service in $services) {
    try {
        $agentJson = azd ai agent show $service 2>$null | ConvertFrom-Json
        $agentPrincipalId = $agentJson.instance_identity.principal_id
    } catch {
        $agentPrincipalId = $null
    }

    if (-not $agentPrincipalId) {
        Write-Host "Could not retrieve identity for $service. Skipping."
        continue
    }

    Write-Host "Assigning Search Index Data Contributor to $service ($agentPrincipalId)..."
    az role assignment create `
        --assignee-object-id $agentPrincipalId `
        --assignee-principal-type ServicePrincipal `
        --role "8ebe5a00-799e-43f5-93ac-243d3dce84a7" `
        --scope $searchScope `
        --only-show-errors `
        --output none 2>$null

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Role assignment may already exist for $service (this is OK)."
    }
}

Write-Host "Postdeploy hook complete."
