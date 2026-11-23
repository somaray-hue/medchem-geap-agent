---
name: pubchem-database
description: Query PubChem to resolve chemical/drug names or SMILES into verified CIDs, canonical SMILES, InChIKeys, and physicochemical descriptors while respecting rate limits. Use whenever verifying a public compound identity.
license: Apache-2.0
---

# PubChem Database Cheminformatics Skill

Adapted from Google DeepMind's Agentic Cheminformatics Science Skills pattern (`google-deepmind/science-skills`).

## Core Rules
1. **Verify Facts Deterministically**: NEVER rely on internal LLM memory for a compound's canonical SMILES, InChIKey, or molecular weight. ALWAYS call `resolve_compound_identifiers_from_pubchem`.
2. **Respect Rate Limits**: Use the rate-limited tool wrapper (`<= 4 req/s`). Do not issue raw unthrottled HTTP loops.
3. **Enterprise IP Fence**: NEVER query PubChem with proprietary internal identifiers (`INTERNAL-*`, `COMP-CONFIDENTIAL-*`) or confidential ELN SMILES.
4. **Fallback Strategy**: If a salt form (e.g., `crizotinib hydrochloride`) fails to resolve, strip the counter-ion and retry with the neutral parent name (`crizotinib`).
5. **Reference**: Consult `references/endpoints.md` for PUG-REST identifier resolution semantics.
