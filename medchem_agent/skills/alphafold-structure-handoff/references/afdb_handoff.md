# AlphaFold 3 & Molecular Dynamics Handoff Specification

When preparing RDKit-validated small molecules for downstream AlphaFold 3 ligand-protein co-folding or free-energy perturbation (FEP) / molecular dynamics:
- **Protein Input**: Verified UniProt ID + AlphaFold DB accession (`AF-<UNIPROT>-F1`) + target kinase domain residue boundaries + resistance mutations (`L1196M`, `G1202R`).
- **Ligand Input**: RDKit-sanitized isomeric canonical SMILES + SDF export generated via `export_validated_synthesis_candidates_to_eln` (after human medicinal-chemist confirmation).
- **Confidence Check**: Always evaluate pLDDT / ipTM binding-pocket confidence experimentally before advancing designs.
