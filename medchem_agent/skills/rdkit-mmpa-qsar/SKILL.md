---
name: rdkit-mmpa-qsar
description: Execute deterministic RDKit molecular standardization, CNS-MPO & PAINS profiling, Matched Molecular Pair Analysis (MMPA) for conformational rigidification, and leak-free Bemis-Murcko scaffold-split QSAR modeling.
license: Apache-2.0
---

# RDKit Cheminformatics, MMPA & QSAR Skill

Encodes Google DeepMind's Nov 2026 Agentic Cheminformatics workflow ("Rigidifying a floppy molecule"):

## Mandatory Workflow Steps
1. **Standardize First**: Pass all raw compounds through `standardize_and_compute_molecular_descriptors` to strip counter-ions, canonicalize SMILES, detect macrocycles (`max_ring_size >= 12`), score Pfizer CNS-MPO (target >= 4.0 for brain-penetrant oncology drugs), and flag PAINS alerts.
2. **Matched Molecular Pair Analysis (MMPA)**: Call `run_matched_molecular_pair_rigidification_analysis` to isolate how cyclization changes rotatable bonds (`ΔRotBonds`), cellular/biochemical potency (`ΔpChEMBL`), and CNS-MPO (`ΔCNS-MPO`).
3. **Leak-Free Molecular ML**: Call `train_scaffold_split_potency_qsar_model` using `split_strategy="scaffold"` (Bemis-Murcko) or `"chronological"`. Verify `scaffold_overlap_count == 0` under scaffold splits.
4. **Validate Every Design**: Call `validate_and_score_candidate_macrocycle_design` for every novel analogue before presenting it. Never output unverified SMILES.
5. **Reference**: Consult `references/split_hygiene.md` before interpreting QSAR RMSE.
