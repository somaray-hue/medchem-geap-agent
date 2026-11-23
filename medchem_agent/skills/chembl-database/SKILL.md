---
name: chembl-database
description: Query ChEMBL for target bioactivities (IC50, Ki, EC50, pChEMBL), filter salts/buffers/solvents, and prepare curated SAR datasets for downstream RDKit MMPA and QSAR modeling.
license: Apache-2.0
---

# ChEMBL Bioactivity Curation Skill

Adapted from Google DeepMind's Agentic Cheminformatics Science Skills pattern (`google-deepmind/science-skills`).

## Core Rules
1. **Standardized Potency Units**: Prefer `pChEMBL` (`-log10(Molar IC50/Ki)`) for cross-compound potency comparisons and QSAR modeling.
2. **Noise & Artifact Hygiene**: Exclude inorganic buffers, salts, solvents, and fragments (`heavy_atoms < 10`), and explicitly report inter-lab assay variance when combining multiple ChEMBL documents.
3. **Provenance Logging**: Always cite the `target_chembl_id` (e.g., `CHEMBL4247` for human ALK), source mode (`chembl_live` vs `chembl_curated_cache`), and record count.
4. **Reference**: Consult `references/bioactivity_rules.md` for assay-type filtering guidance.
