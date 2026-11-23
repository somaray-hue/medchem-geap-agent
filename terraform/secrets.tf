# Google Cloud Secret Manager containers (secret values are populated out-of-band, never in IaC/Git)
locals {
  managed_secrets = toset([
    "pubchem-api-key",
    "eln-api-token",
    "session-db-url",
  ])
}

resource "google_secret_manager_secret" "agent_secrets" {
  for_each  = local.managed_secrets
  secret_id = each.value

  labels = {
    app         = "medchem-geap-agent"
    environment = var.environment
  }

  replication {
    auto {}
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret_iam_member" "agent_secret_accessor" {
  for_each = local.managed_secrets

  project   = var.project_id
  secret_id = google_secret_manager_secret.agent_secrets[each.value].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.medchem_agent_sa.email}"
}
