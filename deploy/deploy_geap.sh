#!/usr/bin/env bash
# Deploy MedChem GEAP Enterprise Agent to Google Cloud Gemini Enterprise Agent Platform (GEAP)
# Supports both ADK CLI (`adk deploy agent_engine`) and Terraform (`google_vertex_ai_reasoning_engine`).
set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID before deploying}"
GCP_LOCATION="${GCP_LOCATION:-us-central1}"
STAGING_BUCKET="${STAGING_BUCKET:-gs://${GCP_PROJECT_ID}-medchem-agent-staging}"
DEPLOY_MODE="${1:-adk}"

echo "==> Running pre-deployment unit tests & golden chemistry regression evals..."
.venv/bin/pytest -v tests/
.venv/bin/python eval/run_eval.py

if [[ "${DEPLOY_MODE}" == "adk" ]]; then
  echo "==> Deploying via Google ADK CLI to GEAP Agent Runtime (Vertex AI Agent Engine)..."
  .venv/bin/adk deploy agent_engine \
    --project="${GCP_PROJECT_ID}" \
    --region="${GCP_LOCATION}" \
    --staging_bucket="${STAGING_BUCKET}" \
    --display_name="MedChem GEAP Enterprise Agent" \
    --trace_to_cloud \
    medchem_agent
elif [[ "${DEPLOY_MODE}" == "terraform" ]]; then
  echo "==> Provisioning GEAP infrastructure and Agent Runtime via Terraform..."
  terraform -chdir=terraform init
  terraform -chdir=terraform plan \
    -var="project_id=${GCP_PROJECT_ID}" \
    -var="region=${GCP_LOCATION}" \
    -out=tfplan
  echo "Review tfplan above and run: terraform -chdir=terraform apply tfplan"
else
  echo "Unknown deploy mode '${DEPLOY_MODE}'. Use 'adk' or 'terraform'." >&2
  exit 1
fi
