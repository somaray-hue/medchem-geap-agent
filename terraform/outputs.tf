output "agent_service_account_email" {
  description = "Least-privilege service account email bound to MedChem GEAP Agent."
  value       = google_service_account.medchem_agent_sa.email
}

output "artifact_registry_url" {
  description = "Docker image repository URL in Artifact Registry."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.agent_repo.repository_id}"
}

output "staging_bucket_name" {
  description = "GCS bucket for SDF provenance exports and ADK staging."
  value       = google_storage_bucket.agent_artifacts.name
}

output "cloud_run_service_url" {
  description = "Internal Cloud Run HTTPS endpoint for the FastAPI + ADK Runner service."
  value       = google_cloud_run_v2_service.medchem_agent_service.uri
}

output "geap_agent_runtime_id" {
  description = "Provisioned Vertex AI Reasoning Engine / GEAP Agent Runtime resource name."
  value       = google_vertex_ai_reasoning_engine.medchem_agent_runtime.id
}
