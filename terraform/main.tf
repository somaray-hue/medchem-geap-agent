locals {
  required_services = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudtrace.googleapis.com",
    "dlp.googleapis.com",
    "logging.googleapis.com",
    "modelarmor.googleapis.com",
    "monitoring.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
  ])

  agent_image_uri    = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_name}/medchem-geap-agent:${var.image_tag}"
  enable_model_armor = var.model_armor_template_id != "" ? "true" : "false"
}

resource "google_project_service" "required_apis" {
  for_each           = local.required_services
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# Artifact Registry repository for MedChem GEAP container images
resource "google_artifact_registry_repository" "agent_repo" {
  location      = var.region
  repository_id = var.repository_name
  description   = "Container registry for MedChem GEAP Enterprise Agent."
  format        = "DOCKER"

  depends_on = [google_project_service.required_apis]
}

# Staging & provenance artifact bucket with uniform bucket-level access
resource "google_storage_bucket" "agent_artifacts" {
  name                        = "${var.project_id}-medchem-agent-${var.environment}"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.required_apis]
}

# GEAP Agent Runtime (Vertex AI Reasoning Engine) registration for managed sessions & Memory Bank
resource "google_vertex_ai_reasoning_engine" "medchem_agent_runtime" {
  provider     = google-beta
  project      = var.project_id
  region       = var.region
  display_name = "medchem-geap-agent-${var.environment}"
  description  = "Enterprise Life Sciences Medicinal Chemistry Agent (ADK + DeepMind Cheminformatics + TxGemma)."

  spec {
    service_account = google_service_account.medchem_agent_sa.email
    agent_framework = "google-adk"

    deployment_spec {
      env {
        name  = "APP_ENV"
        value = var.environment
      }
      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "GCP_LOCATION"
        value = var.region
      }
      env {
        name  = "MODEL_FAST"
        value = var.model_fast
      }
      env {
        name  = "MODEL_REASONING"
        value = var.model_reasoning
      }
      env {
        name  = "TXGEMMA_ENDPOINT_ID"
        value = var.txgemma_endpoint_id
      }
      env {
        name  = "ENABLE_MODEL_ARMOR"
        value = local.enable_model_armor
      }
      env {
        name  = "MODEL_ARMOR_TEMPLATE_ID"
        value = var.model_armor_template_id
      }
      env {
        name  = "ENABLE_SECRET_MANAGER"
        value = "true"
      }
      env {
        name  = "ENABLE_CLOUD_DLP"
        value = "true"
      }
      env {
        name  = "ENABLE_CLOUD_TRACE"
        value = "true"
      }
      secret_env {
        name = "ELN_API_TOKEN"
        secret_ref {
          secret  = google_secret_manager_secret.agent_secrets["eln-api-token"].secret_id
          version = "latest"
        }
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_project_iam_member.medchem_agent_roles,
    google_secret_manager_secret_iam_member.agent_secret_accessor,
  ]
}

# Managed Cloud Run service hosting the FastAPI + ADK Runner + DeepMind Cheminformatics Skills
resource "google_cloud_run_v2_service" "medchem_agent_service" {
  name     = "medchem-geap-agent-${var.environment}"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    service_account = google_service_account.medchem_agent_sa.email

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    containers {
      image = local.agent_image_uri

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }

      env {
        name  = "APP_ENV"
        value = var.environment
      }
      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "GCP_LOCATION"
        value = var.region
      }
      env {
        name  = "AGENT_ENGINE_ID"
        value = google_vertex_ai_reasoning_engine.medchem_agent_runtime.name
      }
      env {
        name  = "MODEL_FAST"
        value = var.model_fast
      }
      env {
        name  = "MODEL_REASONING"
        value = var.model_reasoning
      }
      env {
        name  = "TXGEMMA_ENDPOINT_ID"
        value = var.txgemma_endpoint_id
      }
      env {
        name  = "ENABLE_MODEL_ARMOR"
        value = local.enable_model_armor
      }
      env {
        name  = "MODEL_ARMOR_TEMPLATE_ID"
        value = var.model_armor_template_id
      }
      env {
        name  = "ENABLE_SECRET_MANAGER"
        value = "true"
      }
      env {
        name  = "ENABLE_CLOUD_DLP"
        value = "true"
      }
      env {
        name  = "ENABLE_CLOUD_TRACE"
        value = "true"
      }
      env {
        name = "ELN_API_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.agent_secrets["eln-api-token"].secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [
    google_artifact_registry_repository.agent_repo,
    google_secret_manager_secret_iam_member.agent_secret_accessor,
    google_vertex_ai_reasoning_engine.medchem_agent_runtime,
  ]
}
