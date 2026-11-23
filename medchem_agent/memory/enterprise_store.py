"""Persistent Enterprise ELN & Target Product Profile (TPP) Knowledge Store.

Simulates an internal pharmaceutical Electronic Lab Notebook (ELN) and vector-indexed
knowledge base persisted in SQLite (or backed by Vertex AI Search / RAG Engine in GCP).
Proprietary internal compounds stored here are strictly protected by the IP egress fence
in `ChemGuardrailsAndPolicyPlugin` so their SMILES never leave the enterprise perimeter.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medchem_agent.config.settings import AppSettings, get_settings
from medchem_agent.observability.pii_redactor import get_pii_redactor

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _bow_vector(text: str) -> Counter[str]:
    """Build a normalized term-frequency vector for lightweight semantic retrieval."""
    return Counter(_TOKEN_RE.findall(text.lower()))


def _cosine_similarity(vec_a: Counter[str], vec_b: Counter[str]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EnterpriseElnStore:
    """SQLite-persisted internal ELN compound registry, TPP policies, and campaign memory."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        settings: AppSettings = get_settings()
        self._db_path = Path(db_path or settings.eln_db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._seed_default_enterprise_records()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS proprietary_compounds (
                    compound_id TEXT PRIMARY KEY,
                    target_symbol TEXT NOT NULL,
                    smiles TEXT NOT NULL,
                    pchembl_alk REAL,
                    cns_kp_uu REAL,
                    assay_notes TEXT NOT NULL,
                    confidentiality_tier TEXT NOT NULL DEFAULT 'INTERNAL_ONLY',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS target_product_profiles (
                    target_symbol TEXT PRIMARY KEY,
                    uniprot_id TEXT NOT NULL,
                    indication TEXT NOT NULL,
                    min_pchembl REAL NOT NULL,
                    min_cns_mpo REAL NOT NULL,
                    max_rotatable_bonds INTEGER NOT NULL,
                    resistance_mutations TEXT NOT NULL,
                    alphafold_accession TEXT NOT NULL,
                    notes TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS campaign_notes (
                    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    target_symbol TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def _seed_default_enterprise_records(self) -> None:
        """Seed representative NSCLC ALK/ROS1 enterprise benchmark series & TPP."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            existing_tpp = conn.execute(
                "SELECT COUNT(*) FROM target_product_profiles"
            ).fetchone()[0]
            if existing_tpp == 0:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO target_product_profiles
                    (target_symbol, uniprot_id, indication, min_pchembl, min_cns_mpo,
                     max_rotatable_bonds, resistance_mutations, alphafold_accession, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "ALK",
                            "Q9UM73",
                            "ALK-positive metastatic NSCLC with CNS brain metastases",
                            8.5,
                            4.0,
                            3,
                            "L1196M (gatekeeper), G1202R (solvent front), C1156Y",
                            "AF-Q9UM73-F1",
                            (
                                "Prioritize macrocyclic rigidification to lower entropic binding "
                                "penalty, maintain MW < 430 Da and TPSA <= 115 A^2 for CNS penetrance."
                            ),
                        ),
                        (
                            "ROS1",
                            "P08922",
                            "ROS1-rearranged NSCLC with CNS involvement",
                            8.3,
                            4.0,
                            3,
                            "G2032R (solvent front), D2033N",
                            "AF-P08922-F1",
                            (
                                "Compact macrocyclic hinge binders required to avoid steric clash "
                                "with G2032R while keeping P-gp efflux ratio < 2.5."
                            ),
                        ),
                        (
                            "EGFR",
                            "P00533",
                            "EGFR-mutant NSCLC (Exon19del / L858R / T790M / C797S)",
                            8.0,
                            3.5,
                            5,
                            "T790M, C797S",
                            "AF-P00533-F1",
                            (
                                "4th-generation reversible or allosteric inhibitors sparing wild-type "
                                "EGFR by > 20-fold selectivity."
                            ),
                        ),
                    ],
                )

            existing_comp = conn.execute(
                "SELECT COUNT(*) FROM proprietary_compounds"
            ).fetchone()[0]
            if existing_comp == 0:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO proprietary_compounds
                    (compound_id, target_symbol, smiles, pchembl_alk, cns_kp_uu,
                     assay_notes, confidentiality_tier, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "INTERNAL-ALK-001",
                            "ALK",
                            "C[C@H]1Oc2cc(cnc2N)-c2c(nn(C)c2C#N)CN(C)C(=O)c2ccc(F)cc21",
                            9.35,
                            0.72,
                            "Internal cellular Ba/F3 ALK-WT & L1196M benchmark macrocycle.",
                            "INTERNAL_ONLY",
                            now,
                        ),
                        (
                            "INTERNAL-ALK-002",
                            "ALK",
                            "C[C@H]1Oc2cc(cnc2N)-c2c(nn(C)c2Cl)CN(C)C(=O)c2ccc(F)cc21",
                            8.82,
                            0.64,
                            "Chloro-pyrazole macrocycle backup series; reduced cyano-mediated metabolic liability.",
                            "INTERNAL_ONLY",
                            now,
                        ),
                        (
                            "INTERNAL-ALK-003",
                            "ALK",
                            "C[C@@H](Oc1cc(-c2cnn(C)c2C#N)cnc1N)c1cc(F)ccc1C(=O)NC",
                            7.65,
                            0.28,
                            "Acyclic matched-pair precursor prior to ring-closing amidation.",
                            "INTERNAL_ONLY",
                            now,
                        ),
                    ],
                )

    def get_tpp(self, target_symbol: str) -> dict[str, Any] | None:
        """Fetch Target Product Profile and AlphaFold metadata for a biological target."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM target_product_profiles WHERE UPPER(target_symbol) = UPPER(?)",
                (target_symbol.strip(),),
            ).fetchone()
            return dict(row) if row else None

    def get_proprietary_compounds(self, target_symbol: str) -> list[dict[str, Any]]:
        """Retrieve internal proprietary ELN compounds for a target."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM proprietary_compounds WHERE UPPER(target_symbol) = UPPER(?)",
                (target_symbol.strip(),),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_proprietary_smiles_set(self) -> set[str]:
        """Return all proprietary SMILES registered in the internal ELN for egress checks."""
        with self._connect() as conn:
            rows = conn.execute("SELECT smiles FROM proprietary_compounds").fetchall()
            return {row["smiles"] for row in rows}

    def save_campaign_note(
        self,
        campaign_id: str,
        target_symbol: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Persist a PII-redacted medicinal chemistry campaign insight."""
        redactor = get_pii_redactor()
        safe_summary = redactor.redact_text(summary)
        safe_meta = redactor.redact_payload(metadata or {})
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO campaign_notes (campaign_id, target_symbol, summary, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (campaign_id, target_symbol.upper(), safe_summary, json.dumps(safe_meta), now),
            )
            return int(cursor.lastrowid or 0)

    def search_campaign_notes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantic cosine-similarity search across stored campaign design notes."""
        q_vec = _bow_vector(query)
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM campaign_notes").fetchall()

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            doc_text = f"{row['campaign_id']} {row['target_symbol']} {row['summary']} {row['metadata_json']}"
            sim = _cosine_similarity(q_vec, _bow_vector(doc_text))
            if sim > 0.0:
                entry = dict(row)
                entry["metadata"] = json.loads(entry.pop("metadata_json"))
                entry["relevance_score"] = round(sim, 4)
                scored.append((sim, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]


_default_eln_store: EnterpriseElnStore | None = None


def get_enterprise_eln_store() -> EnterpriseElnStore:
    """Return the process-wide EnterpriseElnStore singleton."""
    global _default_eln_store
    if _default_eln_store is None:
        _default_eln_store = EnterpriseElnStore()
    return _default_eln_store
