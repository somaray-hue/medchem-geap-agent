# Molecular Machine Learning Split Hygiene

- **Why Random Splits Fail in Drug Discovery**: Random splits place near-identical congeneric analogues into both train and test sets, artificially inflating R² and hiding out-of-distribution failure on new chemical series.
- **Chronological Split**: Train on earlier publication years, evaluate on newer analogues to mirror prospective lead optimization.
- **Bemis-Murcko Scaffold Split**: Partition compounds by core ring-linker scaffold (`MurckoScaffold.GetScaffoldForMol`) so `train_scaffolds ∩ test_scaffolds == ∅` (`scaffold_overlap_count == 0`).
- **Rigidification Principle**: Ring closure ("paying the entropic cost in advance") reduces conformational freedom (`ΔRotBonds < 0`), frequently improving binding potency and CNS permeability when MW and TPSA remain controlled.
