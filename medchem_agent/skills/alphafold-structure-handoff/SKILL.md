---
name: alphafold-structure-handoff
description: Prepare protein target context, kinase resistance mutation profiles, and AlphaFold DB / AlphaFold 3 structural handoff packages for validated small-molecule designs.
license: Apache-2.0
---

# AlphaFold Structural Handoff Skill

Connects medicinal chemistry lead optimization with downstream structural biology:

## Core Rules
1. Retrieve the enterprise Target Product Profile and AlphaFold DB accession (e.g., `AF-Q9UM73-F1` for ALK, `AF-P08922-F1` for ROS1) via `retrieve_internal_eln_series_and_tpp`.
2. Document clinically relevant binding-pocket resistance mutations (e.g., gatekeeper `L1196M`, solvent-front `G1202R`) that constrain ligand steric bulk.
3. Ensure candidate ligands exported for AlphaFold 3 co-folding or molecular dynamics carry RDKit-sanitized canonical isomeric SMILES with explicit stereochemistry (`@`/`@@`).
4. Consult `references/afdb_handoff.md` for required ligand-receptor handoff fields.
