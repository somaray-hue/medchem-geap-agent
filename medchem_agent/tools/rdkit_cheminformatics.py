"""Deterministic RDKit cheminformatics, MMPA rigidification, and scaffold-split QSAR tools.

Implements Google DeepMind's Nov 2026 Agentic Cheminformatics methodology:
- All molecular structure standardization, descriptor calculations, PAINS filters, and
  CNS-MPO scoring are executed deterministically in RDKit (zero LLM estimation).
- Matched Molecular Pair Analysis (MMPA) quantifies conformational rigidification
  (cyclization Δrotatable bonds, ΔpChEMBL potency, ΔCNS-MPO).
- QSAR potency modeling enforces Bemis-Murcko scaffold or chronological splits with
  explicit verification that ``scaffold_overlap_count == 0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import ValidationError
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from medchem_agent.observability.tracing import get_tracer
from medchem_agent.schemas.chemistry import (
    CandidateDesignInput,
    CandidateDesignOutput,
    CompoundDescriptorRecord,
    MmpaAnalysisInput,
    MmpaAnalysisOutput,
    MmpaPairRecord,
    QsarModelInput,
    QsarModelOutput,
    StandardizeMoleculesInput,
    StandardizeMoleculesOutput,
    ToolErrorResponse,
)

tracer = get_tracer("medchem_agent.tools.rdkit")

# Initialize PAINS filter catalog & Morgan fingerprint generator once
_PAINS_PARAMS = FilterCatalog.FilterCatalogParams()
_PAINS_PARAMS.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog.FilterCatalog(_PAINS_PARAMS)
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
_LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()


def _piecewise_monotonic(value: float, ideal_max: float, zero_at: float) -> float:
    """Monotonic decreasing desirability function for Pfizer CNS-MPO components."""
    if value <= ideal_max:
        return 1.0
    if value >= zero_at:
        return 0.0
    return (zero_at - value) / (zero_at - ideal_max)


def _piecewise_hump(value: float, low_zero: float, low_one: float, high_one: float, high_zero: float) -> float:
    """Trapezoidal desirability function used for TPSA in Pfizer CNS-MPO."""
    if value <= low_zero or value >= high_zero:
        return 0.0
    if low_one <= value <= high_one:
        return 1.0
    if value < low_one:
        return (value - low_zero) / (low_one - low_zero)
    return (high_zero - value) / (high_zero - high_one)


def compute_cns_mpo_score(
    mw: float, clogp: float, tpsa: float, hbd: int, estimated_pka: float = 8.0
) -> float:
    """Compute Pfizer CNS Multiparameter Optimization (MPO) score (0.0–6.0).

    Compounds with CNS-MPO >= 4.0 have a significantly higher probability of central
    nervous system exposure and low P-gp efflux liability.
    """
    d_clogp = _piecewise_monotonic(clogp, ideal_max=3.0, zero_at=5.0)
    # Approximate cLogD at pH 7.4 conservatively from cLogP for basic/neutral small molecules
    d_clogd = _piecewise_monotonic(clogp - 0.5, ideal_max=2.0, zero_at=4.0)
    d_mw = _piecewise_monotonic(mw, ideal_max=360.0, zero_at=500.0)
    d_tpsa = _piecewise_hump(tpsa, low_zero=20.0, low_one=40.0, high_one=90.0, high_zero=120.0)
    d_hbd = _piecewise_monotonic(float(hbd), ideal_max=0.5, zero_at=3.5)
    d_pka = _piecewise_monotonic(estimated_pka, ideal_max=8.0, zero_at=10.0)
    return round(d_clogp + d_clogd + d_mw + d_tpsa + d_hbd + d_pka, 2)


def _featurize_molecule(
    compound_id: str,
    raw_smiles: str,
    pchembl_value: float | None = None,
) -> tuple[Chem.Mol, CompoundDescriptorRecord] | None:
    """Strip salts, keep largest parent fragment, and compute RDKit descriptors."""
    raw_mol = Chem.MolFromSmiles(raw_smiles)
    if raw_mol is None:
        return None
    parent_mol = _LARGEST_FRAGMENT_CHOOSER.choose(raw_mol)
    Chem.SanitizeMol(parent_mol)

    canonical_smiles = Chem.MolToSmiles(parent_mol, canonical=True)
    inchikey = Chem.MolToInchiKey(parent_mol)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(parent_mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold_mol, canonical=True) if scaffold_mol else canonical_smiles

    mw = round(float(Descriptors.MolWt(parent_mol)), 2)
    clogp = round(float(Crippen.MolLogP(parent_mol)), 2)
    tpsa = round(float(rdMolDescriptors.CalcTPSA(parent_mol)), 2)
    hbd = int(rdMolDescriptors.CalcNumHBD(parent_mol))
    hba = int(rdMolDescriptors.CalcNumHBA(parent_mol))
    rot = int(rdMolDescriptors.CalcNumRotatableBonds(parent_mol))
    ring_count = int(rdMolDescriptors.CalcNumRings(parent_mol))
    ring_sizes = [len(ring) for ring in parent_mol.GetRingInfo().AtomRings()]
    max_ring_size = max(ring_sizes, default=0)
    is_macrocycle = max_ring_size >= 12

    lipinski_violations = int(mw > 500) + int(clogp > 5.0) + int(hbd > 5) + int(hba > 10)
    cns_mpo = compute_cns_mpo_score(mw=mw, clogp=clogp, tpsa=tpsa, hbd=hbd)

    pains_matches = _PAINS_CATALOG.GetMatches(parent_mol)
    pains_alerts = [entry.GetDescription() for entry in pains_matches]

    record = CompoundDescriptorRecord(
        compound_id=compound_id,
        input_smiles=raw_smiles,
        canonical_smiles=canonical_smiles,
        inchikey=inchikey,
        bemis_murcko_scaffold=scaffold_smiles,
        molecular_weight=mw,
        clogp=clogp,
        tpsa=tpsa,
        hbd=hbd,
        hba=hba,
        rotatable_bonds=rot,
        ring_count=ring_count,
        max_ring_size=max_ring_size,
        is_macrocycle=is_macrocycle,
        lipinski_violations=lipinski_violations,
        cns_mpo_score=cns_mpo,
        pains_alerts=pains_alerts,
        pchembl_value=pchembl_value,
    )
    return parent_mol, record


def _morgan_vector(mol: Chem.Mol) -> np.ndarray:
    fp = _MORGAN_GEN.GetFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def standardize_and_compute_molecular_descriptors(
    molecules: list[dict[str, Any]],
) -> dict[str, Any]:
    """Standardize raw SMILES in RDKit (salt stripping + canonicalization) and compute descriptors.

    Computes exact Molecular Weight, cLogP, TPSA, HBD/HBA, Rotatable Bonds, Bemis-Murcko
    scaffold, Macrocycle detection (ring size >= 12), Lipinski Ro5 violations, Pfizer CNS-MPO
    desirability score, and PAINS alerts.

    Args:
        molecules: List of dicts, each containing ``compound_id`` (str), ``smiles`` (str),
            and optional ``pchembl_value`` (float) and ``publication_year`` (int).

    Returns:
        Dictionary matching ``StandardizeMoleculesOutput`` on success or ``ToolErrorResponse``
        if validation fails or zero molecules are chemically valid.
    """
    with tracer.start_as_current_span("tool.standardize_and_compute_molecular_descriptors") as span:
        span.set_attribute("medchem.input_molecule_count", len(molecules))
        try:
            validated = StandardizeMoleculesInput(molecules=molecules)  # type: ignore[arg-type]
        except ValidationError as exc:
            return ToolErrorResponse(
                error_code="SCHEMA_VALIDATION_ERROR",
                message=f"Invalid molecule input list: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Pass a non-empty list of objects with keys `compound_id` and `smiles` "
                    "(optional `pchembl_value` between 0 and 15)."
                ),
                suggested_next_tool="standardize_and_compute_molecular_descriptors",
                retryable=True,
            ).model_dump(mode="json")

        valid_records: list[CompoundDescriptorRecord] = []
        invalid_entries: list[dict[str, str]] = []

        for entry in validated.molecules:
            try:
                res = _featurize_molecule(
                    compound_id=entry.compound_id,
                    raw_smiles=entry.smiles,
                    pchembl_value=entry.pchembl_value,
                )
            except Exception as exc:
                invalid_entries.append(
                    {"compound_id": entry.compound_id, "reason": f"RDKit sanitization error: {exc}"}
                )
                continue

            if res is None:
                invalid_entries.append(
                    {
                        "compound_id": entry.compound_id,
                        "reason": f"Malformed SMILES string '{entry.smiles}' could not be parsed by RDKit.",
                    }
                )
            else:
                _, desc = res
                valid_records.append(desc)

        if not valid_records:
            return ToolErrorResponse(
                error_code="ALL_SMILES_INVALID",
                message="None of the provided SMILES strings parsed validly in RDKit.",
                recovery_instructions=(
                    "Check SMILES syntax or resolve valid canonical structures first using "
                    "`resolve_compound_identifiers_from_pubchem`."
                ),
                suggested_next_tool="resolve_compound_identifiers_from_pubchem",
                retryable=True,
            ).model_dump(mode="json")

        span.set_attribute("medchem.valid_molecule_count", len(valid_records))
        return StandardizeMoleculesOutput(
            valid_count=len(valid_records),
            invalid_entries=invalid_entries,
            compounds=valid_records,
        ).model_dump(mode="json")


def run_matched_molecular_pair_rigidification_analysis(
    compounds: list[dict[str, Any]],
    min_tanimoto_similarity: float = 0.20,
) -> dict[str, Any]:
    """Perform Matched Molecular Pair Analysis (MMPA) comparing acyclic vs. macrocyclic analogues.

    Isolates the entropic rigidification effect of ring closure ("paying the entropic cost in
    advance") by pairing structurally similar acyclic and macrocyclic compounds and quantifying
    mean ``delta_rotatable_bonds``, ``delta_pchembl`` potency gain, and ``delta_cns_mpo`` shift.

    Args:
        compounds: List of dicts with ``compound_id``, ``smiles``, and measured ``pchembl_value``.
        min_tanimoto_similarity: Minimum Morgan fingerprint Tanimoto similarity (0.05–1.0)
            required to form an acyclic-to-macrocycle matched pair.

    Returns:
        Dictionary matching ``MmpaAnalysisOutput`` or ``ToolErrorResponse`` with recovery steps.
    """
    with tracer.start_as_current_span("tool.run_matched_molecular_pair_rigidification_analysis") as span:
        try:
            validated = MmpaAnalysisInput(
                compounds=compounds,  # type: ignore[arg-type]
                min_tanimoto_similarity=min_tanimoto_similarity,
            )
        except ValidationError as exc:
            return ToolErrorResponse(
                error_code="SCHEMA_VALIDATION_ERROR",
                message=f"Invalid MMPA inputs: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Provide at least 2 compounds with `compound_id`, `smiles`, and numeric `pchembl_value`."
                ),
                suggested_next_tool="fetch_target_bioactivities_from_chembl",
                retryable=True,
            ).model_dump(mode="json")

        acyclic: list[tuple[Any, CompoundDescriptorRecord]] = []
        macrocycles: list[tuple[Any, CompoundDescriptorRecord]] = []

        for item in validated.compounds:
            if item.pchembl_value is None:
                continue
            featurized = _featurize_molecule(item.compound_id, item.smiles, item.pchembl_value)
            if featurized is None:
                continue
            mol, desc = featurized
            fp = _MORGAN_GEN.GetFingerprint(mol)
            if desc.is_macrocycle:
                macrocycles.append((fp, desc))
            else:
                acyclic.append((fp, desc))

        if not acyclic or not macrocycles:
            return ToolErrorResponse(
                error_code="INSUFFICIENT_MATCHED_GROUPS",
                message=(
                    f"Need both acyclic ({len(acyclic)} found) and macrocyclic "
                    f"({len(macrocycles)} found) compounds with pChEMBL values for rigidification MMPA."
                ),
                recovery_instructions=(
                    "Fetch a broader series containing both acyclic leads and cyclized analogues "
                    "using `fetch_target_bioactivities_from_chembl`."
                ),
                suggested_next_tool="fetch_target_bioactivities_from_chembl",
                retryable=True,
            ).model_dump(mode="json")

        pairs: list[MmpaPairRecord] = []
        for ac_fp, ac_desc in acyclic:
            for mc_fp, mc_desc in macrocycles:
                sim = float(DataStructs.TanimotoSimilarity(ac_fp, mc_fp))
                if sim >= validated.min_tanimoto_similarity:
                    assert ac_desc.pchembl_value is not None and mc_desc.pchembl_value is not None
                    d_rot = mc_desc.rotatable_bonds - ac_desc.rotatable_bonds
                    d_pot = round(mc_desc.pchembl_value - ac_desc.pchembl_value, 2)
                    d_mpo = round(mc_desc.cns_mpo_score - ac_desc.cns_mpo_score, 2)
                    pairs.append(
                        MmpaPairRecord(
                            acyclic_id=ac_desc.compound_id,
                            macrocycle_id=mc_desc.compound_id,
                            tanimoto_similarity=round(sim, 3),
                            delta_rotatable_bonds=d_rot,
                            delta_pchembl=d_pot,
                            delta_cns_mpo=d_mpo,
                            transformation_summary=(
                                f"Cyclization ({ac_desc.compound_id} -> {mc_desc.compound_id}): "
                                f"ΔRotBonds={d_rot:+d}, ΔpChEMBL={d_pot:+.2f}, ΔCNS-MPO={d_mpo:+.2f}"
                            ),
                        )
                    )

        if not pairs:
            return ToolErrorResponse(
                error_code="NO_MATCHED_PAIRS_ABOVE_THRESHOLD",
                message=(
                    f"No acyclic/macrocycle pairs met Tanimoto >= {validated.min_tanimoto_similarity}."
                ),
                recovery_instructions=(
                    "Retry `run_matched_molecular_pair_rigidification_analysis` with a lower "
                    "`min_tanimoto_similarity` (e.g., 0.15) to detect scaffold-hopping cyclizations."
                ),
                suggested_next_tool="run_matched_molecular_pair_rigidification_analysis",
                retryable=True,
            ).model_dump(mode="json")

        pairs.sort(key=lambda p: (p.tanimoto_similarity, p.delta_pchembl), reverse=True)
        mean_d_rot = round(float(np.mean([p.delta_rotatable_bonds for p in pairs])), 2)
        mean_d_pot = round(float(np.mean([p.delta_pchembl for p in pairs])), 2)
        mean_d_mpo = round(float(np.mean([p.delta_cns_mpo for p in pairs])), 2)

        insight = (
            f"Across {len(pairs)} matched acyclic-to-macrocycle pairs, cyclization reduced "
            f"rotatable bonds by {mean_d_rot:+.2f} on average while shifting potency by "
            f"{mean_d_pot:+.2f} pChEMBL units (~{10 ** max(mean_d_pot, 0):.1f}x potency gain) "
            f"and CNS-MPO by {mean_d_mpo:+.2f}."
        )
        span.set_attribute("medchem.mmpa_pair_count", len(pairs))
        return MmpaAnalysisOutput(
            total_pairs_analyzed=len(pairs),
            mean_delta_rotatable_bonds=mean_d_rot,
            mean_delta_pchembl=mean_d_pot,
            mean_delta_cns_mpo=mean_d_mpo,
            top_pairs=pairs[:10],
            medicinal_chemistry_insight=insight,
        ).model_dump(mode="json")


def train_scaffold_split_potency_qsar_model(
    compounds: list[dict[str, Any]],
    split_strategy: str = "scaffold",
    candidate_smiles: list[str] | None = None,
) -> dict[str, Any]:
    """Train and benchmark leak-free molecular potency QSAR models using scaffold or chronological splits.

    Encodes DeepMind Cheminformatics ML split hygiene:
    - Never uses leaky random splits for drug discovery series.
    - Benchmarks Morgan-fingerprint + physicochemical descriptor models (RandomForest vs Ridge)
      on held-out Bemis-Murcko scaffolds (or newer publication years) and verifies zero
      scaffold overlap on scaffold splits.

    Args:
        compounds: At least 6 labeled compounds with ``compound_id``, ``smiles``, ``pchembl_value``
            (and ``publication_year`` when ``split_strategy="chronological"``).
        split_strategy: ``"scaffold"`` (Bemis-Murcko scaffold split, default) or ``"chronological"``.
        candidate_smiles: Optional list of candidate SMILES to predict pChEMBL potency for.

    Returns:
        Dictionary matching ``QsarModelOutput`` or ``ToolErrorResponse``.
    """
    with tracer.start_as_current_span("tool.train_scaffold_split_potency_qsar_model") as span:
        span.set_attribute("medchem.split_strategy", split_strategy)
        try:
            validated = QsarModelInput(
                compounds=compounds,  # type: ignore[arg-type]
                split_strategy=split_strategy,  # type: ignore[arg-type]
                candidate_smiles=candidate_smiles or [],
            )
        except ValidationError as exc:
            return ToolErrorResponse(
                error_code="SCHEMA_VALIDATION_ERROR",
                message=f"Invalid QSAR training inputs: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Supply at least 6 labeled compounds with `compound_id`, `smiles`, and `pchembl_value`, "
                    "and set `split_strategy` to 'scaffold' or 'chronological'."
                ),
                suggested_next_tool="fetch_target_bioactivities_from_chembl",
                retryable=True,
            ).model_dump(mode="json")

        rows: list[tuple[np.ndarray, float, str, int]] = []
        for entry in validated.compounds:
            if entry.pchembl_value is None:
                continue
            featurized = _featurize_molecule(entry.compound_id, entry.smiles, entry.pchembl_value)
            if featurized is None:
                continue
            mol, desc = featurized
            fp_vec = _morgan_vector(mol)
            physchem = np.array(
                [
                    desc.molecular_weight / 500.0,
                    desc.clogp / 5.0,
                    desc.tpsa / 140.0,
                    desc.rotatable_bonds / 10.0,
                    1.0 if desc.is_macrocycle else 0.0,
                ],
                dtype=np.float32,
            )
            features = np.concatenate([fp_vec, physchem])
            rows.append(
                (features, float(entry.pchembl_value), desc.bemis_murcko_scaffold, entry.publication_year or 2020)
            )

        if len(rows) < 6:
            return ToolErrorResponse(
                error_code="INSUFFICIENT_LABELED_DATA",
                message=f"Only {len(rows)} valid labeled compounds remained after RDKit sanitization (minimum 6 required).",
                recovery_instructions="Curate more labeled bioactivities from ChEMBL before training a QSAR model.",
                suggested_next_tool="fetch_target_bioactivities_from_chembl",
                retryable=True,
            ).model_dump(mode="json")

        # Partition strictly by Bemis-Murcko scaffold or chronological publication year
        used_strategy = validated.split_strategy
        if used_strategy == "chronological" and len({r[3] for r in rows}) >= 2:
            ordered_indices = sorted(range(len(rows)), key=lambda i: rows[i][3])
            split_idx = max(4, int(len(ordered_indices) * 0.75))
            train_idx = ordered_indices[:split_idx]
            test_idx = ordered_indices[split_idx:]
        else:
            used_strategy = "scaffold"
            scaffold_groups: dict[str, list[int]] = {}
            for idx, r in enumerate(rows):
                scaffold_groups.setdefault(r[2], []).append(idx)
            # Deterministic ordering: largest scaffold clusters first into train, remainder into test
            sorted_groups = sorted(
                scaffold_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])
            )
            if len(sorted_groups) < 2:
                return ToolErrorResponse(
                    error_code="INSUFFICIENT_SCAFFOLD_DIVERSITY",
                    message="All compounds share a single Bemis-Murcko scaffold; scaffold split is impossible.",
                    recovery_instructions=(
                        "Retrieve chemically diverse analogues or retry with `split_strategy='chronological'`."
                    ),
                    suggested_next_tool="train_scaffold_split_potency_qsar_model",
                    retryable=True,
                ).model_dump(mode="json")

            target_train = max(4, int(len(rows) * 0.75))
            train_idx, test_idx = [], []
            for _scaf, idxs in sorted_groups:
                if len(train_idx) < target_train and len(test_idx) < len(rows) - 1:
                    if len(train_idx) + len(idxs) <= len(rows) - 2:
                        train_idx.extend(idxs)
                    else:
                        test_idx.extend(idxs)
                else:
                    test_idx.extend(idxs)
            if not test_idx:
                # Guarantee at least one held-out scaffold group
                _, last_idxs = sorted_groups[-1]
                test_idx = list(last_idxs)
                train_idx = [i for i in range(len(rows)) if i not in set(test_idx)]

        X_train = np.stack([rows[i][0] for i in train_idx])
        y_train = np.array([rows[i][1] for i in train_idx], dtype=np.float32)
        X_test = np.stack([rows[i][0] for i in test_idx])
        y_test = np.array([rows[i][1] for i in test_idx], dtype=np.float32)

        train_scaffolds = {rows[i][2] for i in train_idx}
        test_scaffolds = {rows[i][2] for i in test_idx}
        overlap = len(train_scaffolds & test_scaffolds)

        # Benchmark two model families deterministically
        candidates_models = {
            "MorganFP+PhysChem_Ridge": Ridge(alpha=1.0),
            "MorganFP+PhysChem_RandomForest": RandomForestRegressor(
                n_estimators=100, random_state=42, max_depth=6
            ),
        }
        best_name = ""
        best_model: Any = None
        best_rmse = float("inf")
        best_mae = float("inf")

        for name, model in candidates_models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = float(root_mean_squared_error(y_test, preds))
            mae = float(mean_absolute_error(y_test, preds))
            if rmse < best_rmse:
                best_rmse = rmse
                best_mae = mae
                best_name = name
                best_model = model

        # Score optional candidate SMILES
        candidate_preds: list[dict[str, Any]] = []
        for idx, smi in enumerate(validated.candidate_smiles):
            feat = _featurize_molecule(f"CAND-{idx + 1}", smi)
            if feat is None:
                candidate_preds.append(
                    {"input_smiles": smi, "valid": False, "error": "Invalid SMILES"}
                )
                continue
            mol, desc = feat
            vec = np.concatenate(
                [
                    _morgan_vector(mol),
                    np.array(
                        [
                            desc.molecular_weight / 500.0,
                            desc.clogp / 5.0,
                            desc.tpsa / 140.0,
                            desc.rotatable_bonds / 10.0,
                            1.0 if desc.is_macrocycle else 0.0,
                        ],
                        dtype=np.float32,
                    ),
                ]
            ).reshape(1, -1)
            pred_pchembl = round(float(best_model.predict(vec)[0]), 2)
            pred_ic50_nm = round(float(10 ** (9.0 - pred_pchembl)), 2)
            candidate_preds.append(
                {
                    "canonical_smiles": desc.canonical_smiles,
                    "valid": True,
                    "predicted_pchembl": pred_pchembl,
                    "predicted_ic50_nm": pred_ic50_nm,
                    "cns_mpo_score": desc.cns_mpo_score,
                    "is_macrocycle": desc.is_macrocycle,
                }
            )

        span.set_attribute("medchem.qsar_best_model", best_name)
        span.set_attribute("medchem.qsar_test_rmse", round(best_rmse, 3))
        return QsarModelOutput(
            split_strategy_used=used_strategy,  # type: ignore[arg-type]
            train_size=len(train_idx),
            test_size=len(test_idx),
            unique_train_scaffolds=len(train_scaffolds),
            unique_test_scaffolds=len(test_scaffolds),
            scaffold_overlap_count=overlap,
            best_model_name=best_name,
            test_rmse_pchembl=round(best_rmse, 3),
            test_mae_pchembl=round(best_mae, 3),
            candidate_predictions=candidate_preds,
        ).model_dump(mode="json")


def validate_and_score_candidate_macrocycle_design(
    design_id: str,
    proposed_smiles: str,
    reference_smiles: str,
    proposed_synthesis_route: str,
    target_cns_mpo_min: float = 4.0,
) -> dict[str, Any]:
    """Validate a proposed candidate molecule in RDKit and score it against reference & TPP.

    Every proposed analogue or macrocycle MUST pass this deterministic valence, sanitization,
    PAINS, Tanimoto, and CNS-MPO check before being presented to a medicinal chemist.

    Args:
        design_id: Unique design identifier (e.g. ``"DESIGN-ALK-MC-01"``).
        proposed_smiles: Candidate SMILES string to validate.
        reference_smiles: Reference lead SMILES string (e.g. Crizotinib or Lorlatinib).
        proposed_synthesis_route: Proposed synthetic sequence (e.g. macrolactamization or
            intramolecular Mitsunobu cyclization).
        target_cns_mpo_min: Minimum required Pfizer CNS-MPO desirability score (default 4.0).

    Returns:
        Dictionary conforming to ``CandidateDesignOutput`` or ``ToolErrorResponse``.
    """
    with tracer.start_as_current_span("tool.validate_and_score_candidate_macrocycle_design") as span:
        span.set_attribute("medchem.design_id", design_id)
        try:
            validated = CandidateDesignInput(
                design_id=design_id,
                proposed_smiles=proposed_smiles,
                reference_smiles=reference_smiles,
                proposed_synthesis_route=proposed_synthesis_route,
                target_cns_mpo_min=target_cns_mpo_min,
            )
        except ValidationError as exc:
            return ToolErrorResponse(
                error_code="SCHEMA_VALIDATION_ERROR",
                message=f"Invalid candidate design parameters: {exc.errors()[0]['msg']}",
                recovery_instructions="Provide valid design_id, proposed_smiles, reference_smiles, and synthesis route (>= 10 chars).",
                suggested_next_tool="validate_and_score_candidate_macrocycle_design",
                retryable=True,
            ).model_dump(mode="json")

        cand = _featurize_molecule(validated.design_id, validated.proposed_smiles)
        if cand is None:
            return ToolErrorResponse(
                error_code="INVALID_PROPOSED_SMILES",
                message=f"Proposed SMILES '{validated.proposed_smiles}' failed RDKit valence/sanitization.",
                recovery_instructions=(
                    "Fix ring-closure digits, aromaticity, or valence errors in the proposed SMILES "
                    "and re-run `validate_and_score_candidate_macrocycle_design`. Never report unverified SMILES."
                ),
                suggested_next_tool="validate_and_score_candidate_macrocycle_design",
                retryable=True,
            ).model_dump(mode="json")

        ref = _featurize_molecule("REFERENCE", validated.reference_smiles)
        if ref is None:
            return ToolErrorResponse(
                error_code="INVALID_REFERENCE_SMILES",
                message=f"Reference SMILES '{validated.reference_smiles}' failed RDKit parsing.",
                recovery_instructions="Resolve the reference compound canonical SMILES via `resolve_compound_identifiers_from_pubchem`.",
                suggested_next_tool="resolve_compound_identifiers_from_pubchem",
                retryable=True,
            ).model_dump(mode="json")

        cand_mol, cand_desc = cand
        ref_mol, _ = ref
        sim = round(
            float(
                DataStructs.TanimotoSimilarity(
                    _MORGAN_GEN.GetFingerprint(cand_mol),
                    _MORGAN_GEN.GetFingerprint(ref_mol),
                )
            ),
            3,
        )

        return CandidateDesignOutput(
            design_id=validated.design_id,
            canonical_smiles=cand_desc.canonical_smiles,
            inchikey=cand_desc.inchikey,
            tanimoto_to_reference=sim,
            descriptors=cand_desc,
            meets_tpp_cns_mpo=cand_desc.cns_mpo_score >= validated.target_cns_mpo_min,
            meets_lipinski_ro5=cand_desc.lipinski_violations <= 1,
            has_pains_alerts=len(cand_desc.pains_alerts) > 0,
            synthesis_route_summary=validated.proposed_synthesis_route,
            alphafold_handoff_ready=True,
            hypothesis_disclaimer=(
                "Computational design hypothesis only (RDKit-verified topology). "
                "Experimental synthesis and wet-lab biochemical/cellular validation are mandatory."
            ),
        ).model_dump(mode="json")
