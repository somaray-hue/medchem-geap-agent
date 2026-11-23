variable "project_id" {
  description = "Google Cloud Project ID hosting Gemini Enterprise Agent Platform (GEAP)."
  type        = string
}

variable "region" {
  description = "Google Cloud region for GEAP Agent Runtime, Artifact Registry, and Secret Manager."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment label (staging or production)."
  type        = string
  default     = "staging"

  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "environment must be either 'staging' or 'production'."
  }
}

variable "repository_name" {
  description = "Artifact Registry Docker repository name for agent images."
  type        = string
  default     = "medchem-geap-agents"
}

variable "image_tag" {
  description = "Container image tag built by Cloud Build / CI."
  type        = string
  default     = "latest"
}

variable "model_fast" {
  description = "Strategic FAST-tier Gemini model ID."
  type        = string
  default     = "gemini-3.5-flash"
}

variable "model_reasoning" {
  description = "Strategic REASONING-tier Gemini model ID."
  type        = string
  default     = "gemini-3.1-pro-preview"
}

variable "txgemma_endpoint_id" {
  description = "Optional Vertex AI Model Garden endpoint ID for DeepMind TxGemma."
  type        = string
  default     = ""
}

variable "model_armor_template_id" {
  description = "Optional Google Cloud Model Armor template ID for biosecurity and injection screening."
  type        = string
  default     = ""
}
