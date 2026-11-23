# Cloud Logging metric tracking biosecurity & proprietary IP egress guardrail blocks
resource "google_logging_metric" "guardrail_blocks" {
  name        = "medchem_agent_guardrail_blocks_${var.environment}"
  description = "Counts prompt biosecurity blocks and proprietary SMILES IP egress fence blocks."
  filter      = "jsonPayload.event=~\"guardrails\\.(prompt|ip_fence)\\.blocked\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }

  depends_on = [google_project_service.required_apis]
}

# Cloud Logging metric tracking Intent-vs-Outcome mismatches across deterministic chemistry tools
resource "google_logging_metric" "intent_outcome_mismatch" {
  name        = "medchem_agent_intent_outcome_mismatch_${var.environment}"
  description = "Counts tool executions where actual outcome diverged from recorded agent intent."
  filter      = "jsonPayload.event=\"agent.tool.outcome\" AND jsonPayload.intent_outcome_match=false"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }

  depends_on = [google_project_service.required_apis]
}

# Cloud Monitoring alert policy for guardrail spikes
resource "google_monitoring_alert_policy" "guardrail_block_alert" {
  display_name = "MedChem GEAP Agent — Guardrail Policy Block Spike (${var.environment})"
  combiner     = "OR"

  conditions {
    display_name = "Guardrail block rate > 5 in 5 minutes"
    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.guardrail_blocks.name}\" AND resource.type=\"global\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_SUM"
      }
    }
  }

  depends_on = [google_logging_metric.guardrail_blocks]
}
