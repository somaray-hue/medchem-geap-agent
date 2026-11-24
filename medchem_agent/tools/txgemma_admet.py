"""DeepMind TxGemma ADMET & Toxicity Prediction tool for GEAP Model Garden.

Integrates Google DeepMind's TxGemma therapeutics model deployed on GEAP / Vertex AI
Model Garden (``TXGEMMA_ENDPOINT_ID``) for Therapeutics Data Commons (TDC) endpoints
(BBB permeability, hERG cardiotoxicity, CYP3A4 inhibition, AMES mutagenicity, ClinTox, DILI).

When running locally or in CI without a deployed Vertex AI GPU endpoint, the tool executes
a transparent, RDKit-grounded physicochemical & structural-alert fallback explicitly marked
as ``model_backend="txgemma_physicochemical_fallback"``.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from medchem_agent.config.settings import get_settings
from medchem_agent.observability.tracing import get_tracer
from medchem_agent.schemas.chemistry import (
    AdmetEndpointResult,
    AdmetPredictionInput,
    AdmetPredictionOutput,
    ToolErrorResponse,
)

tracer = get_tracer("medchem_agent.tools.txgemma")

# SMARTS structural alerts for Ames mutagenicity / reactive toxicophores
_MUTAGENIC_SMARTS = [
    ("nitroaromatic", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]c1ccccc1"),
    ("aromatic_amine_unhindered", "[NX3;H2]c1ccccc1"),
    ("epoxide", "C1OC1"),
    ("michael_acceptor", "C=CC(=O)[#6,#7,#8]"),
    ("acyl_halide", "C(=O)[Cl,Br,I]"),
]
_COMPILED_ALERTS = [
    (label, Chem.MolFromSmarts(smarts)) for label, smarts in _MUTAGENIC_SMARTS
]

_TDC_PROMPT_TEMPLATES: dict[str, str] = {
    "BBB_Martins": (
        "Instructions: Predict blood-brain barrier permeability for the drug SMILES.\n"
        "Drug SMILES: {smiles}\nAnswer (A=permeable, B=impermeable):"
    ),
    "hERG": (
        "Instructions: Predict hERG potassium channel cardiotoxicity blocking risk.\n"
        "Drug SMILES: {smiles}\nAnswer (A=non-blocker, B=blocker):"
    ),
    "CYP3A4_Veith": (
        "Instructions: Predict CYP3A4 metabolic enzyme inhibition.\n"
        "Drug SMILES: {smiles}\nAnswer (A=non-inhibitor, B=inhibitor):"
    ),
    "AMES": (
        "Instructions: Predict Ames bacterial mutagenicity.\n"
        "Drug SMILES: {smiles}\nAnswer (A=non-mutagenic, B=mutagenic):"
    ),
    "ClinTox": (
        "Instructions: Predict clinical trial toxicity risk.\n"
        "Drug SMILES: {smiles}\nAnswer (A=low risk, B=high risk):"
    ),
    "DILI": (
        "Instructions: Predict drug-induced liver injury risk.\n"
        "Drug SMILES: {smiles}\nAnswer (A=low risk, B=high risk):"
    ),
}


def _call_vertex_txgemma(
    project_id: str,
    location: str,
    endpoint_id: str,
    canonical_smiles: str,
    endpoints: list[str],
) -> list[AdmetEndpointResult] | None:
    """Invoke a deployed DeepMind TxGemma HTTPS endpoint on GEAP / Vertex AI Model Garden."""
    try:
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        token = creds.token
    except Exception:
        return None

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
    )
    instances = [
        {"prompt": _TDC_PROMPT_TEMPLATES[ep].format(smiles=canonical_smiles), "max_tokens": 32}
        for ep in endpoints
    ]

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"instances": instances},
            )
        if resp.status_code != 200:
            return None

        predictions = resp.json().get("predictions", [])
        results: list[AdmetEndpointResult] = []
        for ep, pred in zip(endpoints, predictions, strict=False):
            text = str(pred).strip().upper()
            favorable = text.startswith("(A)") or text.startswith("A")
            results.append(
                AdmetEndpointResult(
                    endpoint=ep,
                    prediction_label="FAVORABLE" if favorable else "HIGH_RISK",
                    probability_score=0.82 if favorable else 0.76,
                    mechanistic_rationale=f"TxGemma Model Garden prediction output: {str(pred)[:180]}",
                )
            )
        return results if results else None
    except Exception:
        return None


def _evaluate_physicochemical_fallback(
    mol: Chem.Mol,
    endpoints: list[str],
) -> list[AdmetEndpointResult]:
    """Deterministic RDKit-based ADMET triage baseline used when no Vertex TxGemma endpoint is set."""
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = int(rdMolDescriptors.CalcNumHBD(mol))
    rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    basic_n = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(NS=O)]")))

    matched_alerts = [
        label for label, pat in _COMPILED_ALERTS if pat is not None and mol.HasSubstructMatch(pat)
    ]

    results: list[AdmetEndpointResult] = []
    for ep in endpoints:
        if ep == "BBB_Martins":
            # Compact TPSA, moderate cLogP, low HBD, and rigidification favor CNS/BBB entry
            score = max(
                0.05,
                min(
                    0.96,
                    0.90
                    - max(0.0, (tpsa - 90.0) / 150.0)
                    - max(0.0, (mw - 420.0) / 400.0)
                    - max(0.0, (hbd - 2) * 0.12)
                    - max(0.0, (rot - 5) * 0.04),
                ),
            )
            label = "FAVORABLE" if score >= 0.60 else ("MODERATE_RISK" if score >= 0.40 else "HIGH_RISK")
            rationale = (
                f"MW={mw:.1f}, TPSA={tpsa:.1f} A^2, HBD={hbd}, RotBonds={rot}; "
                + (
                    "physicochemical profile favors passive BBB permeability."
                    if label == "FAVORABLE"
                    else "elevated polarity/MW/flexibility may limit CNS penetration or increase P-gp efflux."
                )
            )
        elif ep == "hERG":
            # Basic lipophilic amines carry elevated hERG blocking risk
            risk = max(0.05, min(0.95, 0.15 + max(0.0, logp - 3.5) * 0.18 + (0.20 if basic_n >= 2 else 0.0)))
            label = "HIGH_RISK" if risk >= 0.65 else ("MODERATE_RISK" if risk >= 0.40 else "FAVORABLE")
            score = round(1.0 - risk if label == "FAVORABLE" else risk, 2)
            rationale = (
                f"cLogP={logp:.2f}, basic_nitrogens={basic_n}; "
                + ("elevated lipophilic amine risk for hERG channel block." if risk >= 0.40 else "low lipophilic hERG liability.")
            )
        elif ep == "CYP3A4_Veith":
            risk = max(0.08, min(0.92, 0.18 + max(0.0, mw - 420.0) / 350.0 + max(0.0, logp - 3.2) * 0.14))
            label = "HIGH_RISK" if risk >= 0.65 else ("MODERATE_RISK" if risk >= 0.40 else "FAVORABLE")
            score = round(1.0 - risk if label == "FAVORABLE" else risk, 2)
            rationale = f"MW={mw:.1f}, cLogP={logp:.2f}; CYP3A4 active-site lipophilic binding estimate."
        elif ep == "AMES":
            if matched_alerts:
                label = "HIGH_RISK" if "nitroaromatic" in matched_alerts or "epoxide" in matched_alerts else "MODERATE_RISK"
                score = 0.78 if label == "HIGH_RISK" else 0.52
                rationale = f"Structural alert(s) detected: {', '.join(matched_alerts)}."
            else:
                label = "FAVORABLE"
                score = 0.88
                rationale = "No canonical Ames mutagenic structural alerts (nitroaromatic/epoxide/acyl halide) detected."
        else:  # ClinTox / DILI
            risk = max(0.08, min(0.90, 0.15 + (0.25 if matched_alerts else 0.0) + max(0.0, logp - 4.0) * 0.15))
            label = "HIGH_RISK" if risk >= 0.65 else ("MODERATE_RISK" if risk >= 0.40 else "FAVORABLE")
            score = round(1.0 - risk if label == "FAVORABLE" else risk, 2)
            rationale = f"Composite lipophilicity (cLogP={logp:.2f}) and reactive-group ({len(matched_alerts)} alerts) screen."

        results.append(
            AdmetEndpointResult(
                endpoint=ep,
                prediction_label=label,  # type: ignore[arg-type]
                probability_score=round(float(score), 2),
                mechanistic_rationale=rationale,
            )
        )
    return results


def predict_txgemma_admet_and_toxicity_profile(
    compound_id: str,
    smiles: str,
    endpoints: list[Literal["BBB_Martins", "hERG", "CYP3A4_Veith", "AMES", "ClinTox", "DILI"]] | None = None,
) -> dict[str, Any]:
    """Predict ADMET and toxicity endpoints using Google DeepMind TxGemma on GEAP Model Garden.

    Evaluates Therapeutics Data Commons (TDC) tasks including BBB permeability (``BBB_Martins``),
    hERG cardiotoxicity (``hERG``), CYP3A4 inhibition (``CYP3A4_Veith``), Ames mutagenicity
    (``AMES``), Clinical Toxicity (``ClinTox``), and DILI.

    Args:
        compound_id: Identifier for the compound under review (e.g. ``"CHEMBL3286830"``).
        smiles: Chemical structure in SMILES notation (validated in RDKit before inference).
        endpoints: List of TDC endpoints to evaluate; defaults to
            ``["BBB_Martins", "hERG", "CYP3A4_Veith", "AMES", "ClinTox"]``.

    Returns:
        Dictionary conforming to ``AdmetPredictionOutput`` or ``ToolErrorResponse``.
    """
    with tracer.start_as_current_span("tool.predict_txgemma_admet_and_toxicity_profile") as span:
        span.set_attribute("medchem.compound_id", compound_id)
        try:
            validated = AdmetPredictionInput(
                compound_id=compound_id,
                smiles=smiles,
                endpoints=endpoints  # type: ignore[arg-type]
                or ["BBB_Martins", "hERG", "CYP3A4_Veith", "AMES", "ClinTox"],
            )
        except ValidationError as exc:
            return ToolErrorResponse(
                error_code="SCHEMA_VALIDATION_ERROR",
                message=f"Invalid TxGemma ADMET request: {exc.errors()[0]['msg']}",
                recovery_instructions=(
                    "Use valid TDC endpoints from {'BBB_Martins', 'hERG', 'CYP3A4_Veith', 'AMES', 'ClinTox', 'DILI'} "
                    "with a non-empty compound_id and SMILES string."
                ),
                suggested_next_tool="predict_txgemma_admet_and_toxicity_profile",
                retryable=True,
            ).model_dump(mode="json")

        mol = Chem.MolFromSmiles(validated.smiles)
        if mol is None:
            return ToolErrorResponse(
                error_code="INVALID_SMILES",
                message=f"SMILES '{validated.smiles}' failed RDKit parsing prior to TxGemma scoring.",
                recovery_instructions=(
                    "Canonicalize or verify the SMILES via `standardize_and_compute_molecular_descriptors` "
                    "before requesting TxGemma ADMET scoring."
                ),
                suggested_next_tool="standardize_and_compute_molecular_descriptors",
                retryable=True,
            ).model_dump(mode="json")

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        settings = get_settings()

        backend = "txgemma_physicochemical_fallback"
        endpoint_results: list[AdmetEndpointResult] | None = None

        if settings.gcp_project_id and settings.txgemma_endpoint_id:
            endpoint_results = _call_vertex_txgemma(
                project_id=settings.gcp_project_id,
                location=settings.gcp_location,
                endpoint_id=settings.txgemma_endpoint_id,
                canonical_smiles=canonical_smiles,
                endpoints=list(validated.endpoints),
            )
            if endpoint_results is not None:
                backend = "vertex_ai_txgemma_endpoint"

        if endpoint_results is None:
            endpoint_results = _evaluate_physicochemical_fallback(mol, list(validated.endpoints))

        high_risks = [r for r in endpoint_results if r.prediction_label == "HIGH_RISK"]
        mod_risks = [r for r in endpoint_results if r.prediction_label == "MODERATE_RISK"]
        if high_risks:
            overall = "HIGH_RISK"
        elif mod_risks:
            overall = "REVIEW"
        else:
            overall = "PASS"

        span.set_attribute("medchem.txgemma_backend", backend)
        span.set_attribute("medchem.admet_overall_tier", overall)
        return AdmetPredictionOutput(
            model_backend=backend,  # type: ignore[arg-type]
            compound_id=validated.compound_id,
            canonical_smiles=canonical_smiles,
            overall_safety_tier=overall,  # type: ignore[arg-type]
            endpoint_results=endpoint_results,
        ).model_dump(mode="json")
