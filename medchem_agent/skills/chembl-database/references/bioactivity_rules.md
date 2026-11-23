# ChEMBL Bioactivity Curation Best Practices

- **Kinase Targets**:
  - Human ALK: `CHEMBL4247` (UniProt `Q9UM73`)
  - Human ROS1: `CHEMBL5568` (UniProt `P08922`)
  - Human EGFR: `CHEMBL203` (UniProt `P00533`)
- **Potency Conversion**: `pChEMBL = 9 - log10(IC50_nM)` (e.g., `1 nM -> 9.0`, `10 nM -> 8.0`, `100 nM -> 7.0`).
- **Caveat**: Inter-laboratory cellular vs. biochemical IC50 values often exhibit 0.3–0.6 log-unit assay noise; treat model outputs as ranked hypotheses.
