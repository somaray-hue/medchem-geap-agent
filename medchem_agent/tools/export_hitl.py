"""High-stakes Human-in-the-Loop (HITL) tools & asynchronous campaign memory tool.

Demonstrates ADK's native Human-in-the-Loop Action Confirmation mechanisms:
1. Static confirmation: ``FunctionTool(export_validated_synthesis_candidates_to_eln, require_confirmation=True)``
   combined with explicit ``tool_context.request_confirmation(hint=..., payload=...)``
   so no synthesis work order or SDF is exported to the Enterprise ELN without medicinal-chemist sign-off.
2. Dynamic threshold confirmation: ``FunctionTool(submit_external_cro_assay_order, require_confirmation=requires_high_stakes_cro_confirmation)``
   which pauses execution whenever estimated synthesis/assay cost exceeds ``hitl_cost_threshold_usd``
   or batch size exceeds ``hitl_max_auto_compounds``.
3. Non-blocking async memory commit: ``commit_design_decision_to_campaign_memory`` schedules
   background PII-scrubbed indexing without blocking the chemist UI.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from google.adk.tools import FunctionTool, ToolContext
from pydantic import ValidationError
from rdkit import Chem

from medchem_agent.config.settings import get_settings
from medchem_agent.memory.async_consolidator import get_memory_consolidator
from medchem_agent.observability.logging import get_logger
from medchem_agent.observability.pii_redactor import get_pii_redactor
from medchem_agent.observability.tracing import get_tracer
from medchem_agent.schemas.chemistry import (
    CampaignMemoryCommitInput,
    CampaignMemoryCommitOutput,
    CroAssayOrderInput,
    CroAssayOrderOutput,
    ElnExportInput,
    ElnExportOutput,
    ToolErrorResponse,
)

logger = get_logger("tools.export_hitl")
tracer = get_tracer("medchem_agent.tools.hitl")


def export_validated_synthesis_candidates_to_eln(
    campaign_id: str,
    target_symbol: str,
    candidate_smiles_list: list[str],
    estimated_synthesis_cost_usd: float,
    rationale: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Export RDKit-validated synthesis candidates as an SDF + provenance manifest to the ELN.

    HIGH-STAKES ACTION: Requires explicit human medicinal-chemist confirmation before writing
    artifacts or issuing a wet-lab synthesis work order. Uses ADK ``tool_context.request_confirmation``.

    Args:
        campaign_id: Enterprise campaign code (e.g. ``"ALK-MACRO-2026-Q4"``).
        target_symbol: Target gene symbol (e.g. ``"ALK"``).
        candidate_smiles_list: List of RDKit-verified canonical SMILES strings (1–25).
        estimated_synthesis_cost_usd: Estimated chemistry synthesis budget in USD.
        rationale: Scientific rationale summarizing MMPA, QSAR, and TxGemma ADMET justification.
        tool_context: ADK ``ToolContext`` injected automatically by the ADK runtime for HITL confirmation.

    Returns:
        Dictionary matching ``ElnExportOutput`` (``pending_confirmation``, ``exported``, or ``rejected``)
        or ``ToolErrorResponse`` if any candidate SMILES fails RDKit verification.
    """
    with tracer.start_as_current_span("tool.export_validated_synthesis_candidates_to_eln") as span:
        span.set_attribute("medchem.campaign_id", campaign_id)
        try:
            validated = ElnExportInput(
                campaign_id=campaign_id,
                target_symbol=target_symbol,
                candidate_smiles_list=candidate_smiles_list,
                estimated_synthesis_cost_usd=estimated_synthesis_cost_usd,
                rationale=rationale,
            )
        except ValidationError as exc:
            return ToolErrorResponse(
                error_code="SCHEMA_VALIDATION_ERROR",
                message=f"Invalid ELN export payload: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Ensure campaign_id (>= 3 chars), target_symbol, 1-25 valid SMILES, "
                    "non-negative cost, and rationale (>= 10 chars) are supplied."
                ),
                suggested_next_tool="export_validated_synthesis_candidates_to_eln",
                retryable=True,
            ).model_dump(mode="json")

        # Mandatory deterministic RDKit validation gate prior to human review
        verified_mols: list[tuple[str, Chem.Mol]] = []
        for idx, smi in enumerate(validated.candidate_smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return ToolErrorResponse(
                    error_code="UNVERIFIED_SMILES_BLOCKED",
                    message=f"Candidate #{idx + 1} ('{smi}') failed RDKit parsing and cannot be exported to ELN.",
                    recovery_instructions=(
                        "Validate every candidate via `validate_and_score_candidate_macrocycle_design` "
                        "before invoking ELN export."
                    ),
                    suggested_next_tool="validate_and_score_candidate_macrocycle_design",
                    retryable=True,
                ).model_dump(mode="json")
            can_smi = Chem.MolToSmiles(mol, canonical=True)
            mol.SetProp("_Name", f"{validated.campaign_id}-CAND-{idx + 1:02d}")
            mol.SetProp("Canonical_SMILES", can_smi)
            mol.SetProp("Target", validated.target_symbol.upper())
            verified_mols.append((can_smi, mol))

        # Explicit Human-in-the-Loop (HITL) Confirmation Gate via ADK ToolContext (fail closed)
        if tool_context is None:
            logger.warning(
                "hitl.confirmation.missing_context",
                campaign_id=validated.campaign_id,
                compound_count=len(verified_mols),
            )
            return ElnExportOutput(
                status="pending_confirmation",
                campaign_id=validated.campaign_id,
                exported_compound_count=0,
                audit_message=(
                    "ELN export blocked (fail-closed): missing ADK ToolContext required for "
                    "explicit human medicinal-chemist confirmation."
                ),
            ).model_dump(mode="json")

        confirmation = tool_context.tool_confirmation
        if confirmation is None:
            tool_context.request_confirmation(
                hint=(
                    f"APPROVAL REQUIRED: Export {len(verified_mols)} candidate structure(s) for "
                    f"campaign '{validated.campaign_id}' (target {validated.target_symbol.upper()}, "
                    f"est. synthesis cost ${validated.estimated_synthesis_cost_usd:,.2f}) to Enterprise ELN?"
                ),
                payload={
                    "campaign_id": validated.campaign_id,
                    "target_symbol": validated.target_symbol.upper(),
                    "candidate_smiles": [s for s, _ in verified_mols],
                    "estimated_synthesis_cost_usd": validated.estimated_synthesis_cost_usd,
                    "approved": False,
                },
            )
            logger.info(
                "hitl.confirmation.requested",
                campaign_id=validated.campaign_id,
                compound_count=len(verified_mols),
                estimated_cost_usd=validated.estimated_synthesis_cost_usd,
            )
            return ElnExportOutput(
                status="pending_confirmation",
                campaign_id=validated.campaign_id,
                exported_compound_count=0,
                audit_message=(
                    "Execution paused awaiting explicit medicinal-chemist approval via ADK HITL hook."
                ),
            ).model_dump(mode="json")

        if not confirmation.confirmed:
            logger.info(
                "hitl.confirmation.rejected",
                campaign_id=validated.campaign_id,
            )
            return ElnExportOutput(
                status="rejected",
                campaign_id=validated.campaign_id,
                exported_compound_count=0,
                audit_message=(
                    "ELN export aborted: human medicinal-chemist reviewer rejected the confirmation request."
                ),
            ).model_dump(mode="json")

        # Confirmed: write SDF + PII-redacted provenance manifest
        settings = get_settings()
        export_dir = settings.export_dir
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_campaign = "".join(c if c.isalnum() or c in "-_" else "_" for c in validated.campaign_id)
        sdf_path = export_dir / f"{safe_campaign}_{stamp}.sdf"
        manifest_path = export_dir / f"{safe_campaign}_{stamp}_provenance.json"

        writer = Chem.SDWriter(str(sdf_path))
        for _smi, mol in verified_mols:
            writer.write(mol)
        writer.close()

        redactor = get_pii_redactor()
        manifest = {
            "campaign_id": validated.campaign_id,
            "target_symbol": validated.target_symbol.upper(),
            "exported_at_utc": datetime.now(UTC).isoformat(),
            "estimated_synthesis_cost_usd": validated.estimated_synthesis_cost_usd,
            "rationale": redactor.redact_text(validated.rationale),
            "compounds": [
                {
                    "id": mol.GetProp("_Name"),
                    "canonical_smiles": can_smi,
                    "inchikey": Chem.MolToInchiKey(mol),
                }
                for can_smi, mol in verified_mols
            ],
            "hitl_confirmed": True,
            "disclaimer": "Research Use Only — experimental wet-lab validation required.",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        logger.info(
            "hitl.eln_export.completed",
            campaign_id=validated.campaign_id,
            compound_count=len(verified_mols),
            sdf_path=str(sdf_path),
        )
        return ElnExportOutput(
            status="exported",
            campaign_id=validated.campaign_id,
            exported_compound_count=len(verified_mols),
            sdf_artifact_path=str(sdf_path),
            provenance_manifest_path=str(manifest_path),
            audit_message=(
                f"Exported {len(verified_mols)} RDKit-verified candidate(s) to Enterprise ELN "
                "after human approval."
            ),
        ).model_dump(mode="json")


def requires_high_stakes_cro_confirmation(
    compound_count: int,
    estimated_cost_usd: float,
    has_high_risk_admet_flag: bool = False,
) -> bool:
    """Dynamic ADK HITL confirmation predicate for external CRO wet-lab assay orders.

    Triggers mandatory human confirmation whenever cost exceeds ``hitl_cost_threshold_usd``,
    batch size exceeds ``hitl_max_auto_compounds``, or TxGemma flagged any ``HIGH_RISK`` alert.
    """
    settings = get_settings()
    return (
        estimated_cost_usd > settings.hitl_cost_threshold_usd
        or compound_count > settings.hitl_max_auto_compounds
        or has_high_risk_admet_flag
    )


def submit_external_cro_assay_order(
    campaign_id: str,
    cro_vendor: str,
    assay_panel: str,
    compound_count: int,
    estimated_cost_usd: float,
    has_high_risk_admet_flag: bool = False,
) -> dict[str, Any]:
    """Queue a wet-lab biochemical/cellular assay order with an external CRO partner.

    Protected by dynamic ADK HITL confirmation (``requires_high_stakes_cro_confirmation``):
    high-cost orders (> $5,000), batches > 5 compounds, or HIGH_RISK ADMET candidates pause
    for human confirmation before submission.

    Args:
        campaign_id: Internal campaign identifier.
        cro_vendor: Approved Contract Research Organization name (e.g. ``"Eurofins"``).
        assay_panel: Requested assay panel (e.g. ``"ALK WT + L1196M + G1202R IC50 + MDCK-MDR1"``).
        compound_count: Number of compounds in the assay batch (1–50).
        estimated_cost_usd: Estimated CRO purchase order cost in USD.
        has_high_risk_admet_flag: True if TxGemma flagged any HIGH_RISK ADMET/toxicity liability.

    Returns:
        Structured order confirmation dictionary matching ``CroAssayOrderOutput`` or ``ToolErrorResponse``.
    """
    with tracer.start_as_current_span("tool.submit_external_cro_assay_order") as span:
        span.set_attribute("medchem.campaign_id", campaign_id)
        try:
            validated = CroAssayOrderInput(
                campaign_id=campaign_id,
                cro_vendor=cro_vendor,
                assay_panel=assay_panel,
                compound_count=compound_count,
                estimated_cost_usd=estimated_cost_usd,
                has_high_risk_admet_flag=has_high_risk_admet_flag,
            )
        except ValidationError as exc:
            return ToolErrorResponse(
                error_code="INVALID_CRO_ORDER",
                message=f"Invalid CRO assay order payload: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Provide campaign_id (>= 3 chars), cro_vendor, assay_panel, compound_count between 1 and 50, "
                    "and non-negative estimated_cost_usd."
                ),
                suggested_next_tool="submit_external_cro_assay_order",
                retryable=True,
            ).model_dump(mode="json")

        span.set_attribute("medchem.cro_vendor", validated.cro_vendor)
        span.set_attribute("medchem.compound_count", validated.compound_count)
        logger.info(
            "cro_assay_order.queued",
            campaign_id=validated.campaign_id,
            cro_vendor=validated.cro_vendor,
            compound_count=validated.compound_count,
            estimated_cost_usd=validated.estimated_cost_usd,
        )
        return CroAssayOrderOutput(
            campaign_id=validated.campaign_id,
            cro_vendor=validated.cro_vendor,
            assay_panel=validated.assay_panel,
            compound_count=validated.compound_count,
            estimated_cost_usd=validated.estimated_cost_usd,
            disclaimer="Order queued in sandbox mode; no live external CRO spend committed.",
        ).model_dump(mode="json")


async def commit_design_decision_to_campaign_memory(
    campaign_id: str,
    target_symbol: str,
    decision_summary: str,
    key_canonical_smiles: list[str] | None = None,
) -> dict[str, Any]:
    """Schedule non-blocking asynchronous indexing of a medicinal chemistry design decision.

    Dispatches PII-scrubbed memory consolidation onto a background ``asyncio`` task so
    expensive indexing never blocks the chemist's interactive response stream.

    Args:
        campaign_id: Campaign identifier (e.g. ``"ALK-MACRO-2026-Q4"``).
        target_symbol: Drug target gene symbol (e.g. ``"ALK"``).
        decision_summary: Concise summary of SAR insights, TPP updates, or selected designs.
        key_canonical_smiles: Optional list of RDKit-verified canonical SMILES to attach.

    Returns:
        Immediate acknowledgement dictionary matching ``CampaignMemoryCommitOutput`` or ``ToolErrorResponse``.
    """
    with tracer.start_as_current_span("tool.commit_design_decision_to_campaign_memory") as span:
        span.set_attribute("medchem.campaign_id", campaign_id)
        try:
            validated = CampaignMemoryCommitInput(
                campaign_id=campaign_id,
                target_symbol=target_symbol,
                decision_summary=decision_summary,
                key_canonical_smiles=key_canonical_smiles or [],
            )
        except ValidationError as exc:
            return ToolErrorResponse(
                error_code="SCHEMA_VALIDATION_ERROR",
                message=f"Invalid campaign memory commit payload: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Provide campaign_id (>= 3 chars), target_symbol (>= 2 chars), and a meaningful "
                    "decision_summary of at least 10 characters."
                ),
                suggested_next_tool="commit_design_decision_to_campaign_memory",
                retryable=True,
            ).model_dump(mode="json")

        consolidator = get_memory_consolidator()
        job_id = consolidator.schedule_campaign_note_indexing(
            campaign_id=validated.campaign_id,
            target_symbol=validated.target_symbol,
            summary=validated.decision_summary,
            metadata={"canonical_smiles": validated.key_canonical_smiles},
        )
        span.set_attribute("memory.job_id", job_id)
        return CampaignMemoryCommitOutput(
            memory_job_id=job_id,
            campaign_id=validated.campaign_id,
            target_symbol=validated.target_symbol.upper(),
            message=(
                "Campaign decision queued for non-blocking PII-scrubbed background indexing."
            ),
        ).model_dump(mode="json")


# ADK FunctionTool wrappers with native Human-in-the-Loop confirmation policies
export_validated_synthesis_candidates_tool = FunctionTool(
    export_validated_synthesis_candidates_to_eln,
    require_confirmation=True,
)

submit_external_cro_assay_tool = FunctionTool(
    submit_external_cro_assay_order,
    require_confirmation=requires_high_stakes_cro_confirmation,
)
