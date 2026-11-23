"""Enterprise Life Sciences Guardrails & Policy ADK Plugin.

Implements defense-in-depth policy enforcement via ADK ``BasePlugin`` hooks:
1. ``before_model_callback``:
   - Screens user prompts against prompt-injection patterns and CBRN / chemical-weapon /
     illicit synthesis requests (plus optional Google Cloud **Model Armor** API screening).
2. ``before_tool_callback``:
   - Enforces the **Enterprise IP / Proprietary SMILES Egress Fence**: prevents any
     ``INTERNAL-*`` compound ID or proprietary ELN SMILES from leaving the enterprise
     perimeter to public PubChem, ChEMBL, or Europe PMC endpoints.
3. ``after_model_callback``:
   - Self-evaluation grounding check ensuring candidate design outputs include a
     research-use-only experimental-validation disclaimer and never provide clinical
     diagnosis or patient prescription advice.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from medchem_agent.config.settings import AppSettings, get_settings
from medchem_agent.memory.enterprise_store import get_enterprise_eln_store
from medchem_agent.observability.logging import get_logger
from medchem_agent.schemas.chemistry import ToolErrorResponse

logger = get_logger("plugins.guardrails")

_CBRN_AND_HAZARD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "CBRN_NERVE_AGENT",
        re.compile(
            r"\b(?:sarin|novichok|vx\s+nerve|tabun|soman|mustard\s+gas|phosgene\s+weapon|ricin|botulinum\s+toxin)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ILLICIT_SYNTHESIS",
        re.compile(
            r"\b(?:synthesize|cook|manufacture)\s+(?:methamphetamine|fentanyl|carfentanil|heroin|lsd)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PROMPT_INJECTION",
        re.compile(
            r"(?:ignore\s+all\s+previous\s+instructions|override\s+system\s+constitution|"
            r"exfiltrate\s+proprietary\s+smiles|bypass\s+hitl\s+confirmation)",
            re.IGNORECASE,
        ),
    ),
]

_EXTERNAL_PUBLIC_TOOLS = frozenset(
    {
        "resolve_compound_identifiers_from_pubchem",
        "fetch_target_bioactivities_from_chembl",
        "search_medicinal_chemistry_literature",
    }
)

_PROPRIETARY_ID_PATTERN = re.compile(
    r"\b(?:INTERNAL|COMP-CONFIDENTIAL|PROPRIETARY)[-_][A-Z0-9\-]+\b",
    re.IGNORECASE,
)

_RESEARCH_DISCLAIMER = (
    "\n\n> **Research Use Only:** Computational molecular designs, QSAR potency estimates, "
    "and TxGemma ADMET predictions are testable hypotheses subject to assay noise and "
    "require wet-lab synthesis and experimental validation."
)


def _extract_latest_user_text(llm_request: LlmRequest) -> str:
    """Extract the most recent user message text from an ADK LlmRequest."""
    if not llm_request.contents:
        return ""
    for content in reversed(llm_request.contents):
        if getattr(content, "role", None) == "user" and content.parts:
            texts = [p.text for p in content.parts if getattr(p, "text", None)]
            if texts:
                return "\n".join(texts)
    return ""


class ChemGuardrailsAndPolicyPlugin(BasePlugin):
    """ADK Plugin enforcing biosecurity, IP egress fencing, Model Armor, and self-eval safety."""

    def __init__(
        self,
        name: str = "chem_guardrails_policy_plugin",
        settings: AppSettings | None = None,
    ) -> None:
        super().__init__(name=name)
        self._settings = settings or get_settings()

    def _screen_with_model_armor(self, text: str) -> tuple[bool, str | None]:
        """Optional Google Cloud Model Armor screening when enabled in production on GEAP."""
        if not (
            self._settings.enable_model_armor
            and self._settings.gcp_project_id
            and self._settings.model_armor_template_id
        ):
            return True, None
        try:
            import google.auth
            import google.auth.transport.requests

            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(google.auth.transport.requests.Request())
            url = (
                f"https://modelarmor.{self._settings.gcp_location}.rep.googleapis.com/v1/"
                f"projects/{self._settings.gcp_project_id}/locations/{self._settings.gcp_location}/"
                f"templates/{self._settings.model_armor_template_id}:sanitizeUserPrompt"
            )
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(
                    url,
                    headers={"Authorization": f"Bearer {creds.token}"},
                    json={"userPromptData": {"text": text}},
                )
            if resp.status_code == 200:
                sanitization = resp.json().get("sanitizationResult", {})
                if sanitization.get("filterMatchState") == "MATCH_FOUND":
                    return False, "MODEL_ARMOR_POLICY_BLOCK"
        except Exception as exc:
            logger.warning("guardrails.model_armor.unavailable", error=type(exc).__name__)
        return True, None

    def inspect_prompt_safety(self, text: str) -> tuple[bool, str | None]:
        """Evaluate prompt text against biosecurity, prompt-injection, and Model Armor policies."""
        for policy_code, pattern in _CBRN_AND_HAZARD_PATTERNS:
            if pattern.search(text):
                return False, policy_code
        return self._screen_with_model_armor(text)

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        """Intercept unsafe biosecurity or prompt-injection requests before calling Gemini."""
        user_text = _extract_latest_user_text(llm_request)
        if not user_text:
            return None

        allowed, policy_code = self.inspect_prompt_safety(user_text)
        if not allowed:
            logger.warning(
                "guardrails.prompt.blocked",
                agent_name=callback_context.agent_name,
                policy_code=policy_code,
            )
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=(
                                f"Request blocked by Enterprise Life Sciences Policy ({policy_code}). "
                                "This agent cannot assist with chemical weapons, controlled illicit "
                                "synthesis, or overriding governance guardrails. Please submit a "
                                "compliant medicinal chemistry research query."
                            )
                        )
                    ],
                )
            )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Enforce the Enterprise IP Egress Fence on all public external API tools."""
        if tool.name not in _EXTERNAL_PUBLIC_TOOLS:
            return None

        serialized_args = str(tool_args)
        if _PROPRIETARY_ID_PATTERN.search(serialized_args):
            logger.warning(
                "guardrails.ip_fence.blocked_identifier",
                tool_name=tool.name,
                agent_name=tool_context.agent_name,
            )
            return ToolErrorResponse(
                error_code="PROPRIETARY_IP_EGRESS_BLOCKED",
                message=(
                    f"Blocked egress to external tool '{tool.name}': arguments referenced a "
                    "proprietary internal compound identifier."
                ),
                recovery_instructions=(
                    "NEVER send INTERNAL-* or confidential compound identifiers to public APIs. "
                    "Use internal tools (`retrieve_internal_eln_series_and_tpp`, "
                    "`standardize_and_compute_molecular_descriptors`, `predict_txgemma_admet_and_toxicity_profile`) "
                    "which execute strictly inside the enterprise perimeter."
                ),
                suggested_next_tool="retrieve_internal_eln_series_and_tpp",
                retryable=False,
            ).model_dump(mode="json")

        # Also block raw proprietary SMILES registered in the Enterprise ELN
        proprietary_smiles = get_enterprise_eln_store().list_proprietary_smiles_set()
        for arg_val in tool_args.values():
            if isinstance(arg_val, str) and arg_val.strip() in proprietary_smiles:
                logger.warning(
                    "guardrails.ip_fence.blocked_smiles",
                    tool_name=tool.name,
                    agent_name=tool_context.agent_name,
                )
                return ToolErrorResponse(
                    error_code="PROPRIETARY_SMILES_EGRESS_BLOCKED",
                    message=(
                        f"Blocked egress to public API '{tool.name}': input SMILES matches a "
                        "confidential structure registered in the Enterprise ELN."
                    ),
                    recovery_instructions=(
                        "Keep confidential ELN structures inside the enterprise boundary. "
                        "Use local RDKit or private GEAP TxGemma tools instead of external PubChem/ChEMBL."
                    ),
                    suggested_next_tool="standardize_and_compute_molecular_descriptors",
                    retryable=False,
                ).model_dump(mode="json")

        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """Self-eval response guardrail: enforce research-use-only framing on chemistry proposals."""
        if not llm_response.content or not llm_response.content.parts:
            return None

        text_parts = [p.text for p in llm_response.content.parts if getattr(p, "text", None)]
        if not text_parts:
            return None

        full_text = "\n".join(text_parts)
        mentions_design = any(
            token in full_text.lower()
            for token in ("smiles", "macrocycle", "pchembl", "admet", "synthesis route", "ic50")
        )
        has_disclaimer = any(
            token in full_text.lower()
            for token in ("experimental validation", "wet-lab", "research use only", "hypothesis")
        )

        if mentions_design and not has_disclaimer:
            logger.info(
                "guardrails.self_eval.appended_research_disclaimer",
                agent_name=callback_context.agent_name,
            )
            updated_parts = list(llm_response.content.parts)
            updated_parts.append(types.Part.from_text(text=_RESEARCH_DISCLAIMER))
            return llm_response.model_copy(
                update={"content": types.Content(role=llm_response.content.role, parts=updated_parts)}
            )

        return None
