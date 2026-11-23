# Least-privilege dedicated Service Account for MedChem GEAP Agent Runtime
resource "google_service_account" "medchem_agent_sa" {
  account_id   = "medchem-geap-agent-${var.environment}"
  display_name = "MedChem GEAP Enterprise Agent (${var.environment})"
  description  = "Dedicated least-privilege identity for MedChem GEAP Agent Runtime workloads."
  depends_on   = [google_project_service.required_apis]
}

locals {
  agent_iam_roles = toset([
    "roles/aiplatform.user",
    "roles/logging.logWriter",
    "roles/cloudtrace.agent",
    "roles/monitoring.metricWriter",
    "roles/dlp.user",
    "roles/storage.objectUser",
  ])
}

resource "google_project_iam_member" "medchem_agent_roles" {
  for_each = local.agent_iam_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.medchem_agent_sa.email}"
}
