#!/usr/bin/env python3
"""Deterministic PubChem identifier resolution helper script for ADK SkillToolset."""

from __future__ import annotations

import argparse
import json
import sys

from medchem_agent.tools.pubchem_chembl import resolve_compound_identifiers_from_pubchem


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve a compound via PubChem + RDKit.")
    parser.add_argument("--query", required=True, help="Compound name or SMILES string.")
    parser.add_argument("--query-type", choices=["name", "smiles"], default="name")
    args = parser.parse_args()

    result = resolve_compound_identifiers_from_pubchem(
        query=args.query,
        query_type=args.query_type,
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
