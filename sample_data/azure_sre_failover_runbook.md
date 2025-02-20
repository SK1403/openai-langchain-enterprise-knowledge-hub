# Azure & Databricks SRE Disaster Recovery Runbook
Runbook ID: SRE-AZURE-DR-2025-01
Owner: Global Azure Infrastructure Engineering Team
Classification: Internal Critical Runbook

## 1. Outage Classification & Microsoft Entra ID Alerting
- **Severity-1 Emergency**: Azure East US 2 region outage affecting Azure OpenAI or Azure Databricks workspace.
- **Incident Commander**: Page on-duty Azure Principal SRE via PagerDuty (`az-infra-tier1`).
- **War Room Bridge**: Join Microsoft Teams War Room channel `SRE-Emergency-Response`.

## 2. Regional Failover Procedure (East US 2 -> West US 3)
In the event of an unrecoverable regional disruption exceeding 15 minutes:

1. **Traffic Manager / Azure Front Door Rerouting**:
   ```bash
   az network front-door backend-pool backend update \
       --resource-group rg-enterprise-ai-platform-prod \
       --front-door-name afd-contoso-core \
       --pool-name default-pool \
       --index 1 \
       --enabled-state Enabled
   ```

2. **Activate Azure Databricks Secondary Workspace**:
   - Verify secondary cluster in West US 3 is hydrated from geo-replicated Delta Lake tables on ADLS Gen2 (`RA-GRS`).
   - Run verification notebook `/Production/HealthChecks/verify_delta_integrity`.

3. **Switch Azure OpenAI Endpoint DNS**:
   - Update API Gateway endpoint configuration from `aoai-eastus2.openai.azure.com` to `aoai-westus3.openai.azure.com`.
   - Validate response latency is under 350ms.

## 3. SLA Targets
- **Recovery Time Objective (RTO)**: <= 20 minutes
- **Recovery Point Objective (RPO)**: <= 2 minutes (via Delta Lake asynchronous continuous sync)
