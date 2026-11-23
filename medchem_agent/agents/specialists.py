"""Specialized ADK LlmAgents wired with Strategic Model Routing & DeepMind Science Skills.

Each specialist uses a factory function so ADK's single-parent tree invariant is respected
and each sub-agent is routed to the optimal Gemini model tier via ``ModelRouter``.
"""

from __future__ import annotations

from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset

from medchem_agent.config.model_router import ModelRouter, TaskComplexity
from medchem_agent.prompts.constitution import (
    ADMET_TRIAGE_INSTRUCTION,
    CURATOR_INSTRUCTION,
    SAR_MMPA_MODELING_INSTRUCTION,
    SELF_CRITIC_INSTRUCTION,
    SYNTHESIS_DESIGN_INSTRUCTION,
    TARGET_CONTEXT_INSTRUCTION,
)
from medchem_agent.schemas.chemistry import SelfCritiqueReport
from medchem_agent.tools.export_hitl import (
    commit_design_decision_to_campaign_memory,
    export_validated_synthesis_candidates_tool,
    submit_external_cro_assay_tool,
)
from medchem_agent.tools.pubchem_chembl import (
    fetch_target_bioactivities_from_chembl,
    resolve_compound_identifiers_from_pubchem,
    retrieve_internal_eln_series_and_tpp,
    search_medicinal_chemistry_literature,
)
from medchem_agent.tools.rdkit_cheminformatics import (
    run_matched_molecular_pair_rigidification_analysis,
    standardize_and_compute_molecular_descriptors,
    train_scaffold_split_potency_qsar_model,
    validate_and_score_candidate_macrocycle_design,
)
from medchem_agent.tools.txgemma_admet import predict_txgemma_admet_and_toxicity_profile

SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"


def _build_skill_toolset(skill_names: list[str], prefix: str | None = None) -> SkillToolset:
    """Load DeepMind-pattern SKILL.md directories into an ADK SkillToolset (lazy discovery)."""
    loaded = [load_skill_from_dir(SKILLS_ROOT / name) for name in skill_names]
    return SkillToolset(skills=loaded, tool_name_prefix=prefix)


def build_curator_agent(router: ModelRouter | None = None) -> LlmAgent:
    """Build the FAST-tier Compound & Literature Curation Agent."""
    mr = router or ModelRouter()
    route = mr.route(TaskComplexity.EXTRACTION)
    curation_skills = _build_skill_toolset(["pubchem-database", "chembl-database"])
    return LlmAgent(
        name="compound_and_literature_curator_agent",
        model=route.model_id,
        description=(
            "Curates target bioactivities from ChEMBL, verifies compound identifiers in PubChem, "
            "retrieves Europe PMC medicinal chemistry literature, and loads internal ELN series."
        ),
        instruction=CURATOR_INSTRUCTION,
        tools=[
            curation_skills,
            resolve_compound_identifiers_from_pubchem,
            fetch_target_bioactivities_from_chembl,
            search_medicinal_chemistry_literature,
            retrieve_internal_eln_series_and_tpp,
        ],
        output_key="curated_compound_data",
    )


def build_target_context_agent(router: ModelRouter | None = None) -> LlmAgent:
    """Build the FAST-tier Target & AlphaFold Structural Context Agent."""
    mr = router or ModelRouter()
    route = mr.route(TaskComplexity.EXTRACTION)
    af_skill = _build_skill_toolset(["alphafold-structure-handoff"])
    return LlmAgent(
        name="target_and_structure_context_agent",
        model=route.model_id,
        description=(
            "Retrieves enterprise Target Product Profile (TPP) cutoffs, kinase resistance "
            "mutations, and AlphaFold DB structural handoff metadata."
        ),
        instruction=TARGET_CONTEXT_INSTRUCTION,
        tools=[
            af_skill,
            retrieve_internal_eln_series_and_tpp,
        ],
        output_key="target_structural_context",
    )


def build_sar_mmpa_agent(router: ModelRouter | None = None) -> LlmAgent:
    """Build the REASONING-tier SAR, MMPA Rigidification & Scaffold-Split QSAR Agent."""
    mr = router or ModelRouter()
    route = mr.route(TaskComplexity.HYPOTHESIS_SYNTHESIS)
    rdkit_skill = _build_skill_toolset(["rdkit-mmpa-qsar"])
    return LlmAgent(
        name="sar_mmpa_modeling_agent",
        model=route.model_id,
        description=(
            "Executes deterministic RDKit descriptor standardization, Matched Molecular Pair "
            "Analysis (MMPA) for rigidification, and Bemis-Murcko scaffold-split QSAR modeling."
        ),
        instruction=SAR_MMPA_MODELING_INSTRUCTION,
        tools=[
            rdkit_skill,
            standardize_and_compute_molecular_descriptors,
            run_matched_molecular_pair_rigidification_analysis,
            train_scaffold_split_potency_qsar_model,
        ],
        output_key="sar_mmpa_analysis",
    )


def build_admet_triage_agent(router: ModelRouter | None = None) -> LlmAgent:
    """Build the FAST-tier + DeepMind TxGemma ADMET & Toxicity Triage Agent."""
    mr = router or ModelRouter()
    route = mr.route(TaskComplexity.TRIAGE)
    return LlmAgent(
        name="admet_safety_triage_agent",
        model=route.model_id,
        description=(
            "Scores lead compounds across TDC ADMET and toxicity endpoints (BBB, hERG, "
            "CYP3A4, AMES, ClinTox) using Google DeepMind TxGemma on GEAP Model Garden."
        ),
        instruction=ADMET_TRIAGE_INSTRUCTION,
        tools=[
            predict_txgemma_admet_and_toxicity_profile,
            standardize_and_compute_molecular_descriptors,
        ],
        output_key="admet_triage_report",
    )


def build_synthesis_design_agent(router: ModelRouter | None = None) -> LlmAgent:
    """Build the REASONING-tier Macrocycle Design, Validation & HITL Export Agent."""
    mr = router or ModelRouter()
    route = mr.route(TaskComplexity.HYPOTHESIS_SYNTHESIS)
    return LlmAgent(
        name="synthesis_and_proposal_agent",
        model=route.model_id,
        description=(
            "Proposes RDKit-validated rigidified analogues with synthesis routes, schedules "
            "async campaign memory commits, and executes HITL-gated ELN exports & CRO orders."
        ),
        instruction=SYNTHESIS_DESIGN_INSTRUCTION,
        tools=[
            validate_and_score_candidate_macrocycle_design,
            export_validated_synthesis_candidates_tool,
            submit_external_cro_assay_tool,
            commit_design_decision_to_campaign_memory,
        ],
        output_key="candidate_design_proposal",
    )


def build_self_critic_agent(router: ModelRouter | None = None) -> LlmAgent:
    """Build the REASONING-tier Scientific Self-Critic with strict Pydantic output schema."""
    mr = router or ModelRouter()
    route = mr.route(TaskComplexity.SELF_CRITIQUE)
    return LlmAgent(
        name="scientific_self_critic_agent",
        model=route.model_id,
        description=(
            "Performs automated pre-delivery scientific self-evaluation verifying RDKit "
            "structural validity, scaffold-split hygiene, and research-use-only disclaimers."
        ),
        instruction=SELF_CRITIC_INSTRUCTION,
        output_schema=SelfCritiqueReport,
        output_key="self_critique_report",
    )


def build_quick_chemistry_lookup_agent(router: ModelRouter | None = None) -> LlmAgent:
    """Build a FAST-tier specialist for single-turn compound, ADMET, or HITL ELN/CRO actions."""
    mr = router or ModelRouter()
    route = mr.route(TaskComplexity.EXTRACTION)
    quick_skills = _build_skill_toolset(["pubchem-database", "rdkit-mmpa-qsar"])
    return LlmAgent(
        name="quick_chemistry_lookup_agent",
        model=route.model_id,
        description=(
            "Handles fast, focused single-compound queries (PubChem lookup, RDKit descriptors, "
            "CNS-MPO/PAINS check, TxGemma ADMET triage, internal ELN TPP retrieval) as well as "
            "standalone HITL-gated Enterprise ELN SDF exports and external CRO assay orders "
            "without triggering the full multi-stage lead-optimization pipeline."
        ),
        instruction=(
            f"{CURATOR_INSTRUCTION}\n\n"
            "Handle concise single-step chemistry queries rapidly using deterministic tools only. "
            "For standalone Enterprise ELN export requests, ALWAYS validate the candidate SMILES "
            "with `validate_and_score_candidate_macrocycle_design` and then invoke "
            "`export_validated_synthesis_candidates_to_eln` (never substitute "
            "`commit_design_decision_to_campaign_memory` for an ELN SDF export). "
            "For external CRO assay orders, invoke `submit_external_cro_assay_order`. "
            "Always include exact canonical SMILES, tool-computed values, and the research-use-only note."
        ),
        tools=[
            quick_skills,
            resolve_compound_identifiers_from_pubchem,
            retrieve_internal_eln_series_and_tpp,
            standardize_and_compute_molecular_descriptors,
            predict_txgemma_admet_and_toxicity_profile,
            validate_and_score_candidate_macrocycle_design,
            export_validated_synthesis_candidates_tool,
            submit_external_cro_assay_tool,
            commit_design_decision_to_campaign_memory,
        ],
    )
