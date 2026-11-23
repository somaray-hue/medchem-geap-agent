"""Strategic Model Routing across Gemini Flash, Gemini Pro, and DeepMind TxGemma.

Routes each specialist agent and subtask to the optimal cost/latency/reasoning tier:
- FAST (`gemini-2.5-flash`): Database curation, literature parsing, target lookups,
  event history compaction, and ADMET triage orchestration.
- REASONING (`gemini-2.5-pro`): Root campaign planning, SAR/MMPA hypothesis synthesis,
  macrocycle design reasoning, and scientific self-evaluation critique.
- SPECIALIST_THERAPEUTICS (`txgemma` Model Garden endpoint): Specialized molecular
  ADMET, BBB permeability, CYP inhibition, and toxicity scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from medchem_agent.config.settings import AppSettings, get_settings


class ModelTier(StrEnum):
    """Named model capability tiers for strategic routing."""

    FAST = "fast"
    REASONING = "reasoning"
    SPECIALIST_THERAPEUTICS = "specialist_therapeutics"


class TaskComplexity(StrEnum):
    """Task complexity classification used for dynamic model selection."""

    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    TRIAGE = "triage"
    PLANNING = "planning"
    HYPOTHESIS_SYNTHESIS = "hypothesis_synthesis"
    SELF_CRITIQUE = "self_critique"
    ADMET_PREDICTION = "admet_prediction"


@dataclass(frozen=True)
class ModelRouteDecision:
    """Auditable model selection record returned by ModelRouter."""

    tier: ModelTier
    model_id: str
    rationale: str


class ModelRouter:
    """Centralized policy router mapping medicinal chemistry workloads to models."""

    _COMPLEXITY_TO_TIER: dict[TaskComplexity, ModelTier] = {
        TaskComplexity.EXTRACTION: ModelTier.FAST,
        TaskComplexity.SUMMARIZATION: ModelTier.FAST,
        TaskComplexity.TRIAGE: ModelTier.FAST,
        TaskComplexity.PLANNING: ModelTier.REASONING,
        TaskComplexity.HYPOTHESIS_SYNTHESIS: ModelTier.REASONING,
        TaskComplexity.SELF_CRITIQUE: ModelTier.REASONING,
        TaskComplexity.ADMET_PREDICTION: ModelTier.SPECIALIST_THERAPEUTICS,
    }

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def model_for_tier(self, tier: ModelTier) -> str:
        """Resolve the configured model identifier for a given tier."""
        if tier == ModelTier.FAST:
            return self._settings.model_fast
        if tier == ModelTier.REASONING:
            return self._settings.model_reasoning
        if tier == ModelTier.SPECIALIST_THERAPEUTICS:
            return self._settings.txgemma_endpoint_id or "txgemma-9b-predict-local-fallback"
        raise ValueError(f"Unsupported model tier: {tier}")

    def route(
        self,
        complexity: TaskComplexity,
        *,
        estimated_input_tokens: int = 0,
        require_high_precision: bool = False,
    ) -> ModelRouteDecision:
        """Select the optimal model tier for a task with an explicit audit rationale.

        Args:
            complexity: Domain task classification.
            estimated_input_tokens: Approximate prompt payload size; large multi-compound
                synthesis tasks automatically escalate to the REASONING tier.
            require_high_precision: Escalates triage/extraction tasks to REASONING when
                high-stakes medicinal chemistry precision is explicitly requested.

        Returns:
            A ``ModelRouteDecision`` specifying the tier, resolved model ID, and rationale.
        """
        tier = self._COMPLEXITY_TO_TIER[complexity]

        if tier == ModelTier.FAST and (require_high_precision or estimated_input_tokens > 32000):
            tier = ModelTier.REASONING
            rationale = (
                f"Escalated '{complexity.value}' to REASONING tier "
                f"(high_precision={require_high_precision}, tokens={estimated_input_tokens})."
            )
        elif tier == ModelTier.FAST:
            rationale = f"Routed '{complexity.value}' to low-latency FAST tier for cost efficiency."
        elif tier == ModelTier.REASONING:
            rationale = f"Routed '{complexity.value}' to high-capability REASONING tier."
        else:
            rationale = (
                "Routed therapeutic property prediction to DeepMind TxGemma specialist model."
            )

        return ModelRouteDecision(
            tier=tier,
            model_id=self.model_for_tier(tier),
            rationale=rationale,
        )
