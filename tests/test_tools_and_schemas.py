"""Tests for Category 1: Tool & Interface Design and deterministic DeepMind chemistry tools."""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from medchem_agent.schemas.chemistry import (
    PubChemResolveInput,
    export_all_tool_json_schemas,
)
from medchem_agent.tools.export_hitl import (
    commit_design_decision_to_campaign_memory,
    export_validated_synthesis_candidates_to_eln,
    submit_external_cro_assay_order,
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

ALL_TOOL_FUNCTIONS = [
    resolve_compound_identifiers_from_pubchem,
    fetch_target_bioactivities_from_chembl,
    search_medicinal_chemistry_literature,
    retrieve_internal_eln_series_and_tpp,
    standardize_and_compute_molecular_descriptors,
    run_matched_molecular_pair_rigidification_analysis,
    train_scaffold_split_potency_qsar_model,
    validate_and_score_candidate_macrocycle_design,
    predict_txgemma_admet_and_toxicity_profile,
    export_validated_synthesis_candidates_to_eln,
    submit_external_cro_assay_order,
    commit_design_decision_to_campaign_memory,
]

CRIZOTINIB_SMILES = "C[C@@H](Oc1cc(-c2cnn(C3CCNCC3)c2)cnc1N)c1c(Cl)ccc(F)c1Cl"
LORLATINIB_SMILES = "C[C@H]1Oc2cc(cnc2N)-c2c(nn(C)c2C#N)CN(C)C(=O)c2ccc(F)cc21"


def test_all_tools_have_comprehensive_docstrings_and_descriptive_names() -> None:
    """Verify every tool has a clear domain-specific name and Args/Returns docstring."""
    for fn in ALL_TOOL_FUNCTIONS:
        assert len(fn.__name__.split("_")) >= 3, f"Tool name too generic: {fn.__name__}"
        doc = inspect.getdoc(fn) or ""
        assert len(doc) >= 120, f"Docstring too brief for {fn.__name__}"
        assert "Args:" in doc and "Returns:" in doc, f"Missing Args/Returns in {fn.__name__}"


def test_strict_pydantic_json_schemas_reject_extra_fields() -> None:
    """Verify Pydantic schemas enforce strict extra='forbid' and export valid JSON Schema for all tools."""
    with pytest.raises(ValidationError):
        PubChemResolveInput(query="lorlatinib", query_type="name", unexpected_param=123)  # type: ignore[call-arg]

    schemas = export_all_tool_json_schemas()
    for expected_schema in (
        "PubChemResolveInput",
        "InternalElnQueryInput",
        "InternalElnQueryOutput",
        "QsarModelOutput",
        "CroAssayOrderInput",
        "CroAssayOrderOutput",
        "CampaignMemoryCommitInput",
        "CampaignMemoryCommitOutput",
        "ToolErrorResponse",
    ):
        assert expected_schema in schemas
        assert schemas[expected_schema]["additionalProperties"] is False


def test_guided_error_handling_returns_recovery_instructions() -> None:
    """Malformed SMILES and invalid inputs must return guided ToolErrorResponse, never crash."""
    bad_smiles = standardize_and_compute_molecular_descriptors(
        [{"compound_id": "BAD-1", "smiles": "NOT_A_VALID_SMILES((("}]
    )
    assert bad_smiles["status"] == "error"
    assert bad_smiles["error_code"] == "ALL_SMILES_INVALID"
    assert "recovery_instructions" in bad_smiles and len(bad_smiles["recovery_instructions"]) > 20
    assert bad_smiles["suggested_next_tool"] == "resolve_compound_identifiers_from_pubchem"

    bad_pubchem = resolve_compound_identifiers_from_pubchem("C1CC(((invalid", query_type="smiles")
    assert bad_pubchem["status"] == "error"
    assert bad_pubchem["error_code"] == "INVALID_SMILES"
    assert bad_pubchem["retryable"] is True

    bad_admet = predict_txgemma_admet_and_toxicity_profile("X1", "invalid_smiles")
    assert bad_admet["status"] == "error"
    assert bad_admet["error_code"] == "INVALID_SMILES"

    # CRO batch cap (> 50 compounds) enforced via CroAssayOrderInput
    bad_cro = submit_external_cro_assay_order(
        campaign_id="ALK-2026-Q4",
        cro_vendor="Eurofins",
        assay_panel="ALK WT IC50",
        compound_count=75,
        estimated_cost_usd=1000.0,
    )
    assert bad_cro["status"] == "error"
    assert bad_cro["error_code"] == "INVALID_CRO_ORDER"
    assert "recovery_instructions" in bad_cro

    bad_eln = retrieve_internal_eln_series_and_tpp("A")
    assert bad_eln["status"] == "error"
    assert bad_eln["error_code"] == "INVALID_TARGET_SYMBOL"


def test_rdkit_standardization_salt_stripping_and_crizotinib_vs_lorlatinib() -> None:
    """Verify deterministic RDKit salt stripping, macrocycle detection, and InChIKey accuracy."""
    # Append HCl salt to crizotinib to confirm salt stripping works deterministically
    result = standardize_and_compute_molecular_descriptors(
        [
            {"compound_id": "CRIZ-HCL", "smiles": f"{CRIZOTINIB_SMILES}.Cl", "pchembl_value": 7.62},
            {"compound_id": "LORL", "smiles": LORLATINIB_SMILES, "pchembl_value": 9.15},
        ]
    )
    assert result["status"] == "success"
    assert result["valid_count"] == 2
    by_id = {c["compound_id"]: c for c in result["compounds"]}

    assert by_id["CRIZ-HCL"]["rotatable_bonds"] == 5
    assert by_id["CRIZ-HCL"]["is_macrocycle"] is False
    assert "." not in by_id["CRIZ-HCL"]["canonical_smiles"]

    assert by_id["LORL"]["rotatable_bonds"] == 0
    assert by_id["LORL"]["is_macrocycle"] is True
    assert by_id["LORL"]["max_ring_size"] == 12
    assert by_id["LORL"]["inchikey"] == "IIXWYSCJSQVBQM-LLVKDONJSA-N"
    assert by_id["LORL"]["cns_mpo_score"] > by_id["CRIZ-HCL"]["cns_mpo_score"]


def test_mmpa_rigidification_and_zero_leakage_scaffold_split_qsar() -> None:
    """Verify MMPA rigidification deltas and Bemis-Murcko zero-leakage QSAR modeling."""
    alk = fetch_target_bioactivities_from_chembl("ALK", "IC50", 25)
    assert alk["status"] == "success"
    compounds = [
        {
            "compound_id": r["compound_id"],
            "smiles": r["canonical_smiles"],
            "pchembl_value": r["pchembl_value"],
            "publication_year": r["publication_year"],
        }
        for r in alk["records"]
    ]

    mmpa = run_matched_molecular_pair_rigidification_analysis(compounds, min_tanimoto_similarity=0.20)
    assert mmpa["status"] == "success"
    assert mmpa["total_pairs_analyzed"] >= 3
    assert mmpa["mean_delta_rotatable_bonds"] < 0
    assert mmpa["mean_delta_pchembl"] > 0

    qsar_scaffold = train_scaffold_split_potency_qsar_model(
        compounds,
        split_strategy="scaffold",
        candidate_smiles=[LORLATINIB_SMILES],
    )
    assert qsar_scaffold["status"] == "success"
    assert qsar_scaffold["split_strategy_used"] == "scaffold"
    assert qsar_scaffold["scaffold_overlap_count"] == 0
    assert len(qsar_scaffold["candidate_predictions"]) == 1
    assert qsar_scaffold["candidate_predictions"][0]["valid"] is True

    qsar_chrono = train_scaffold_split_potency_qsar_model(
        compounds, split_strategy="chronological"
    )
    assert qsar_chrono["status"] == "success"
    assert qsar_chrono["split_strategy_used"] == "chronological"


def test_txgemma_admet_and_mutagenic_alert_detection() -> None:
    """Verify TxGemma ADMET scoring for lorlatinib and high-risk alert flagging for nitrobenzene."""
    lorl_admet = predict_txgemma_admet_and_toxicity_profile("LORL", LORLATINIB_SMILES)
    assert lorl_admet["status"] == "success"
    endpoints = {e["endpoint"]: e for e in lorl_admet["endpoint_results"]}
    assert endpoints["BBB_Martins"]["prediction_label"] == "FAVORABLE"
    assert endpoints["AMES"]["prediction_label"] == "FAVORABLE"

    # Nitrobenzene carries a canonical nitroaromatic Ames alert
    nitro_admet = predict_txgemma_admet_and_toxicity_profile("NITRO-1", "O=[N+]([O-])c1ccccc1", ["AMES"])
    assert nitro_admet["status"] == "success"
    assert nitro_admet["overall_safety_tier"] == "HIGH_RISK"
    assert nitro_admet["endpoint_results"][0]["prediction_label"] == "HIGH_RISK"


def test_candidate_validation_tool() -> None:
    """Verify RDKit candidate design validation scores macrocycle and sets AlphaFold handoff flag."""
    res = validate_and_score_candidate_macrocycle_design(
        design_id="DESIGN-01",
        proposed_smiles=LORLATINIB_SMILES,
        reference_smiles=CRIZOTINIB_SMILES,
        proposed_synthesis_route="Pd-catalyzed C-H arylation followed by macrolactamization.",
    )
    assert res["status"] == "success"
    assert res["alphafold_handoff_ready"] is True
    assert res["has_pains_alerts"] is False
    assert "experimental" in res["hypothesis_disclaimer"].lower()
