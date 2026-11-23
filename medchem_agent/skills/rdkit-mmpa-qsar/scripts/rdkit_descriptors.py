#!/usr/bin/env python3
"""Deterministic RDKit descriptor & CNS-MPO calculation script for ADK SkillToolset."""

from __future__ import annotations

import argparse
import json
import sys

from medchem_agent.tools.rdkit_cheminformatics import (
    standardize_and_compute_molecular_descriptors,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute RDKit descriptors for a SMILES string.")
    parser.add_argument("--compound-id", default="QUERY-1")
    parser.add_argument("--smiles", required=True, help="Input SMILES string.")
    args = parser.parse_args()

    result = standardize_and_compute_molecular_descriptors(
        [{"compound_id": args.compound_id, "smiles": args.smiles}]
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
