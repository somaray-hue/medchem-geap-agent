"""Strict Pydantic v2 input/output JSON schemas for deterministic chemistry tools.

Every tool validates arguments and return payloads against these explicit models
(`extra="forbid"`) so LLMs are strictly constrained and every failure returns a
structured `ToolErrorResponse` with actionable recovery instructions.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    """Base schema rejecting undeclared fields to enforce strict LLM tool contracts."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ToolErrorResponse(StrictSchema):
    """Structured guided error payload returned to the LLM instead of raw exceptions."""

    status: Literal["error"] = "error"
    error_code: str = Field(
        ...,
        description="Machine-readable error identifier (e.g. INVALID_SMILES, RATE_LIMITED).",
    )
    message: str = Field(..., description="Clear human- and LLM-readable explanation of what failed.")
    recovery_instructions: str = Field(
        ...,
        description="Concrete step-by-step instructions telling the agent how to recover.",
    )
    suggested_next_tool: str | None = Field(
        default=None,
        description="Optional recommended follow-up tool to call next.",
    )
    retryable: bool = Field(
        default=False,
        description="Whether retrying with modified or identical arguments is appropriate.",
    )


# ---------------------------------------------------------------------------
# 1. PubChem, ChEMBL, and Literature Curation Schemas
# ---------------------------------------------------------------------------


class PubChemResolveInput(StrictSchema):
    """Input schema for resolving compound names or SMILES in PubChem."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Compound common name (e.g. 'crizotinib', 'lorlatinib') or SMILES string.",
    )
    query_type: Literal["name", "smiles"] = Field(
        default="name",
        description="Whether `query` is a chemical/drug name or a SMILES string.",
    )


class PubChemResolveOutput(StrictSchema):
    """Verified chemical identifiers and properties resolved from PubChem."""

    status: Literal["success"] = "success"
    source: Literal["pubchem_live", "pubchem_curated_cache"] = Field(
        ..., description="Origin of the chemical record."
    )
    query: str
    pubchem_cid: int = Field(..., gt=0)
    canonical_smiles: str
    inchikey: str = Field(..., min_length=27, max_length=27)
    molecular_formula: str
    molecular_weight: float = Field(..., gt=0)
    xlogp: float | None = None
    tpsa: float | None = None
    rotatable_bond_count: int = Field(..., ge=0)
    citation: str = Field(
        default="Kim S et al. PubChem 2025 update. Nucleic Acids Res. 2025."
    )


class ChemblBioactivityInput(StrictSchema):
    """Input schema for querying bioactivity records from ChEMBL."""

    target_query: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="ChEMBL target ID (e.g. 'CHEMBL4247') or gene symbol (e.g. 'ALK', 'EGFR').",
    )
    standard_type: Literal["IC50", "Ki", "EC50", "Kd"] = Field(
        default="IC50",
        description="Bioactivity endpoint to retrieve.",
    )
    max_records: int = Field(
        default=25,
        ge=1,
        le=200,
        description="Maximum curated bioactivity records to return.",
    )


class BioactivityRecord(StrictSchema):
    """Single standardized compound bioactivity entry."""

    compound_id: str
    compound_name: str
    canonical_smiles: str
    standard_type: str
    standard_value_nm: float = Field(..., gt=0)
    pchembl_value: float = Field(..., ge=0.0, le=15.0)
    is_macrocycle: bool
    assay_description: str
    publication_year: int | None = None


class ChemblBioactivityOutput(StrictSchema):
    """Curated bioactivity dataset for a biological target."""

    status: Literal["success"] = "success"
    source: Literal["chembl_live", "chembl_curated_cache"]
    target_chembl_id: str
    target_name: str
    record_count: int = Field(..., ge=0)
    excluded_salts_and_solvents_count: int = Field(..., ge=0)
    records: list[BioactivityRecord]
    provenance_note: str


class LiteratureSearchInput(StrictSchema):
    """Input schema for searching medicinal chemistry literature via Europe PMC."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Scientific search query (e.g. 'lorlatinib macrocycle ALK rigidification').",
    )
    max_results: int = Field(default=5, ge=1, le=20)


class LiteratureArticle(StrictSchema):
    """Single scientific publication metadata record."""

    pmid: str | None = None
    pmcid: str | None = None
    title: str
    journal: str
    year: int
    doi: str | None = None
    abstract_snippet: str


class LiteratureSearchOutput(StrictSchema):
    """Grounded bibliography returned by literature search."""

    status: Literal["success"] = "success"
    source: Literal["europe_pmc_live", "europe_pmc_curated_cache"]
    query: str
    articles: list[LiteratureArticle]


# ---------------------------------------------------------------------------
# 2. Deterministic RDKit Cheminformatics, MMPA & QSAR Schemas
# ---------------------------------------------------------------------------


class MoleculeInputEntry(StrictSchema):
    """Single molecule input with optional measured potency."""

    compound_id: str = Field(..., min_length=1, max_length=64)
    smiles: str = Field(..., min_length=1, max_length=2000)
    pchembl_value: float | None = Field(default=None, ge=0.0, le=15.0)
    publication_year: int | None = Field(default=None, ge=1950, le=2030)


class StandardizeMoleculesInput(StrictSchema):
    """Input schema for RDKit standardization and descriptor calculation."""

    molecules: list[MoleculeInputEntry] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="List of raw molecules to strip salts, canonicalize, and featurize.",
    )


class CompoundDescriptorRecord(StrictSchema):
    """Deterministic RDKit physicochemical, CNS-MPO, and PAINS profile."""

    compound_id: str
    input_smiles: str
    canonical_smiles: str
    inchikey: str
    bemis_murcko_scaffold: str
    molecular_weight: float
    clogp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    ring_count: int
    max_ring_size: int
    is_macrocycle: bool = Field(
        ..., description="True when the compound contains a ring of >= 12 heavy atoms."
    )
    lipinski_violations: int = Field(..., ge=0, le=4)
    cns_mpo_score: float = Field(
        ...,
        ge=0.0,
        le=6.0,
        description="Pfizer CNS Multiparameter Optimization (MPO) desirability score (0-6; >=4 favorable).",
    )
    pains_alerts: list[str]
    pchembl_value: float | None = None


class StandardizeMoleculesOutput(StrictSchema):
    """Output of deterministic RDKit molecular standardization."""

    status: Literal["success"] = "success"
    valid_count: int
    invalid_entries: list[dict[str, str]]
    compounds: list[CompoundDescriptorRecord]


class MmpaAnalysisInput(StrictSchema):
    """Input schema for Matched Molecular Pair rigidification analysis."""

    compounds: list[MoleculeInputEntry] = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Compounds with `smiles` and `pchembl_value` for matched pair comparison.",
    )
    min_tanimoto_similarity: float = Field(
        default=0.20,
        ge=0.05,
        le=1.0,
        description="Minimum Morgan fingerprint Tanimoto similarity to pair acyclic vs macrocyclic analogues.",
    )


class MmpaPairRecord(StrictSchema):
    """Matched molecular pair quantifying rigidification impact on rotatable bonds and potency."""

    acyclic_id: str
    macrocycle_id: str
    tanimoto_similarity: float
    delta_rotatable_bonds: int = Field(
        ..., description="Change in rotatable bonds (macrocycle - acyclic); negative means rigidified."
    )
    delta_pchembl: float = Field(
        ..., description="Change in pChEMBL potency (macrocycle - acyclic); positive means more potent."
    )
    delta_cns_mpo: float = Field(
        ..., description="Change in CNS-MPO desirability score (macrocycle - acyclic)."
    )
    transformation_summary: str


class MmpaAnalysisOutput(StrictSchema):
    """Aggregated MMPA rigidification findings."""

    status: Literal["success"] = "success"
    total_pairs_analyzed: int
    mean_delta_rotatable_bonds: float
    mean_delta_pchembl: float
    mean_delta_cns_mpo: float
    top_pairs: list[MmpaPairRecord]
    medicinal_chemistry_insight: str


class QsarModelInput(StrictSchema):
    """Input schema for training a leak-free molecular potency QSAR model."""

    compounds: list[MoleculeInputEntry] = Field(
        ...,
        min_length=6,
        max_length=500,
        description="Labeled compounds with SMILES and pChEMBL potency.",
    )
    split_strategy: Literal["scaffold", "chronological"] = Field(
        default="scaffold",
        description="Data-hygiene split strategy per DeepMind Cheminformatics best practices (never random).",
    )
    candidate_smiles: list[str] = Field(
        default_factory=list,
        description="Optional novel candidate SMILES to score with the trained QSAR model.",
    )


class QsarModelOutput(StrictSchema):
    """Cross-validated QSAR benchmark metrics and candidate predictions."""

    status: Literal["success"] = "success"
    split_strategy_used: Literal["scaffold", "chronological"]
    train_size: int
    test_size: int
    unique_train_scaffolds: int
    unique_test_scaffolds: int
    scaffold_overlap_count: int = Field(
        ...,
        ge=0,
        description="Must be 0 under Bemis-Murcko scaffold split to prove zero scaffold leakage.",
    )
    best_model_name: str
    test_rmse_pchembl: float
    test_mae_pchembl: float
    candidate_predictions: list[dict[str, Any]]
    caveat: str = Field(
        default=(
            "Computational QSAR predictions are testable hypotheses subject to sparse data "
            "and inter-lab assay noise; experimental synthesis and wet-lab validation are required."
        )
    )


# ---------------------------------------------------------------------------
# 3. DeepMind TxGemma ADMET & Candidate Validation Schemas
# ---------------------------------------------------------------------------


class AdmetPredictionInput(StrictSchema):
    """Input schema for DeepMind TxGemma ADMET & toxicity prediction."""

    compound_id: str = Field(..., min_length=1, max_length=64)
    smiles: str = Field(..., min_length=1, max_length=2000)
    endpoints: list[Literal["BBB_Martins", "hERG", "CYP3A4_Veith", "AMES", "ClinTox", "DILI"]] = (
        Field(
            default_factory=lambda: ["BBB_Martins", "hERG", "CYP3A4_Veith", "AMES", "ClinTox"],
            min_length=1,
            description="Therapeutics Data Commons (TDC) endpoints evaluated via TxGemma.",
        )
    )


class AdmetEndpointResult(StrictSchema):
    """Prediction and mechanistic rationale for a single TDC ADMET endpoint."""

    endpoint: str
    prediction_label: Literal["FAVORABLE", "MODERATE_RISK", "HIGH_RISK"]
    probability_score: float = Field(..., ge=0.0, le=1.0)
    mechanistic_rationale: str


class AdmetPredictionOutput(StrictSchema):
    """Full ADMET and safety triage report from DeepMind TxGemma."""

    status: Literal["success"] = "success"
    model_backend: Literal["vertex_ai_txgemma_endpoint", "txgemma_physicochemical_fallback"]
    compound_id: str
    canonical_smiles: str
    overall_safety_tier: Literal["PASS", "REVIEW", "HIGH_RISK"]
    endpoint_results: list[AdmetEndpointResult]
    disclaimer: str = Field(
        default=(
            "Research Use Only — TxGemma predictions are in-silico hypotheses and do not "
            "constitute clinical or regulatory toxicology clearance."
        )
    )


class CandidateDesignInput(StrictSchema):
    """Input schema for validating and scoring a proposed analogue/macrocycle."""

    design_id: str = Field(..., min_length=1, max_length=64)
    proposed_smiles: str = Field(..., min_length=1, max_length=2000)
    reference_smiles: str = Field(..., min_length=1, max_length=2000)
    proposed_synthesis_route: str = Field(
        ...,
        min_length=10,
        max_length=1500,
        description="Medicinal chemistry reaction strategy (e.g. intramolecular amide/ether cyclization).",
    )
    target_cns_mpo_min: float = Field(default=4.0, ge=0.0, le=6.0)


class CandidateDesignOutput(StrictSchema):
    """RDKit-verified candidate macrocycle evaluation ready for chemist review."""

    status: Literal["success"] = "success"
    design_id: str
    canonical_smiles: str
    inchikey: str
    tanimoto_to_reference: float
    descriptors: CompoundDescriptorRecord
    meets_tpp_cns_mpo: bool
    meets_lipinski_ro5: bool
    has_pains_alerts: bool
    synthesis_route_summary: str
    alphafold_handoff_ready: bool
    hypothesis_disclaimer: str


# ---------------------------------------------------------------------------
# 4. Human-in-the-Loop (HITL) Enterprise ELN Export & Self-Critique Schemas
# ---------------------------------------------------------------------------


class ElnExportInput(StrictSchema):
    """Input schema for exporting verified synthesis candidates to the Enterprise ELN."""

    campaign_id: str = Field(..., min_length=3, max_length=64)
    target_symbol: str = Field(..., min_length=2, max_length=32)
    candidate_smiles_list: list[str] = Field(..., min_length=1, max_length=25)
    estimated_synthesis_cost_usd: float = Field(..., ge=0.0, le=1_000_000.0)
    rationale: str = Field(..., min_length=10, max_length=2000)


class ElnExportOutput(StrictSchema):
    """Result of HITL-gated enterprise ELN SDF/JSON export."""

    status: Literal["exported", "pending_confirmation", "rejected"]
    campaign_id: str
    exported_compound_count: int = Field(..., ge=0)
    sdf_artifact_path: str | None = None
    provenance_manifest_path: str | None = None
    audit_message: str


class SelfCritiqueReport(StrictSchema):
    """Structured output schema for the Scientific Self-Critic evaluation agent."""

    chemical_validity_verified: bool = Field(
        ..., description="True only if all reported structures were validated via RDKit tools."
    )
    deterministic_calculations_enforced: bool = Field(
        ..., description="True only if numeric properties came from tools, not LLM estimation."
    )
    split_hygiene_verified: bool = Field(
        ..., description="True if QSAR used Bemis-Murcko scaffold or chronological split."
    )
    hypothesis_disclaimer_present: bool = Field(
        ..., description="True if output clearly states experimental validation is required."
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    identified_gaps: list[str]
    final_recommendation: str


def export_all_tool_json_schemas() -> dict[str, dict[str, Any]]:
    """Export formal JSON Schema definitions for all strict tool contracts."""
    schema_models = [
        PubChemResolveInput,
        PubChemResolveOutput,
        ChemblBioactivityInput,
        ChemblBioactivityOutput,
        LiteratureSearchInput,
        LiteratureSearchOutput,
        StandardizeMoleculesInput,
        StandardizeMoleculesOutput,
        MmpaAnalysisInput,
        MmpaAnalysisOutput,
        QsarModelInput,
        QsarModelOutput,
        AdmetPredictionInput,
        AdmetPredictionOutput,
        CandidateDesignInput,
        CandidateDesignOutput,
        ElnExportInput,
        ElnExportOutput,
        SelfCritiqueReport,
        ToolErrorResponse,
    ]
    return {model.__name__: model.model_json_schema() for model in schema_models}
