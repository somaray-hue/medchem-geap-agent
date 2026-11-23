# PubChem PUG-REST Reference

- **Name resolution**: `GET /rest/pug/compound/name/{name}/property/CanonicalSMILES,IsomericSMILES,InChIKey,MolecularFormula,MolecularWeight,XLogP,TPSA,RotatableBondCount/JSON`
- **SMILES / InChI resolution**: Always use `HTTP POST` (form field `smiles=...` or `inchi=...`) rather than URL path encoding to avoid slash/hash escaping bugs on aromatic rings and triple bonds (`C#N`).
- **Citation**: Kim S et al. PubChem 2025 update. *Nucleic Acids Research* (2025).
