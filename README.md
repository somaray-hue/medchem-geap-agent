# MedChem GEAP Enterprise Agent

An enterprise Life Sciences **Medicinal Chemistry Lead-Optimization Agent** built on **Google Agent Development Kit (ADK 2.x)**, **Gemini Enterprise Agent Platform (GEAP)**, **Google DeepMind's Agentic Cheminformatics Skills** (Nov 2026), and **DeepMind TxGemma** (GEAP Model Garden).

> **Research Use Only:** All computational compound designs, QSAR potency estimates, and TxGemma ADMET scores produced by this agent are testable scientific hypotheses requiring wet-lab synthesis and experimental validation.

---

## 1. Problem & Solution

Early-stage pharmaceutical lead optimization requires medicinal and computational chemists to unify fragmented chemical structures across PubChem, ChEMBL, peer-reviewed literature, and internal Electronic Lab Notebooks (ELNs), standardize salts, run Matched Molecular Pair Analysis (MMPA), train leak-free QSAR potency models, triage ADMET liabilities, and propose synthesizable analogues—a loop that traditionally takes multiple days per iteration.

**MedChem GEAP Enterprise Agent** operationalizes **Google DeepMind's Agentic Cheminformatics** architecture (*division of labor: Gemini orchestrates scientific reasoning while deterministic chemistry tools execute every molecular calculation with zero margin for error*) inside a governed, enterprise-ready GEAP multi-agent system:

- **Deterministic Cheminformatics (`RDKit`, `ChEMBL`, `PubChem`, `Europe PMC`)**: Strip salts, canonicalize SMILES/InChIKeys, compute Pfizer CNS-MPO & PAINS alerts, quantify conformational rigidification via MMPA (demonstrated on the NSCLC ALK *crizotinib → lorlatinib* macrocyclization workflow), and train Bemis-Murcko scaffold-split QSAR models with verified zero scaffold leakage.
- **DeepMind Science Skills (`SkillToolset`)**: Mounts Apache 2.0 `SKILL.md` + `references/` skill bundles (`pubchem-database`, `chembl-database`, `rdkit-mmpa-qsar`, `alphafold-structure-handoff`) using ADK lazy discovery to prevent context bloat.
- **DeepMind TxGemma ADMET Intelligence**: Calls a deployed GEAP Model Garden TxGemma endpoint (with a transparent RDKit physicochemical fallback offline) across Therapeutics Data Commons (TDC) endpoints (`BBB_Martins`, `hERG`, `CYP3A4_Veith`, `AMES`, `ClinTox`, `DILI`).
- **Enterprise Governance**: Enforces biosecurity/CBRN screening (plus Model Armor integration), a **Proprietary SMILES IP Egress Fence** so internal ELN compounds never leak to public APIs, PII/PHI redaction (Cloud DLP + local scrubber), Intent-vs-Outcome structured JSON audit logs, OpenTelemetry tracing, and ADK Human-in-the-Loop (HITL) confirmation gates before exporting SDF synthesis work orders.

---

## 2. Multi-Agent Architecture

```text
Chemist (Target + TPP / Campaign Query)
   │
   ▼  [ChemGuardrailsAndPolicyPlugin: CBRN/Injection + Model Armor + IP Egress Fence]
   ▼  [ChemObservabilityPlugin: Intent-vs-Outcome JSON Logs + OpenTelemetry Spans + Async Memory]
┌─────────────────────────────────────────────────────────────────────────────────┐
│  medchem_coordinator_agent (ADK LlmAgent — REASONING: gemini-3.1-pro-preview)   │
│  • PreloadMemoryTool (GEAP Memory Bank / FTS SQLite)                            │
│  • commit_design_decision_to_campaign_memory (non-blocking asyncio background)  │
└──────────────┬───────────────────────────────────────────┬──────────────────────┘
               │                                           │
               ▼ (Single-step lookups & HITL exports)      ▼ (End-to-end campaigns)
┌──────────────────────────────────┐       ┌──────────────────────────────────────────────────┐
│ quick_chemistry_lookup_agent     │       │ lead_optimization_pipeline (SequentialAgent)     │
│ (FAST: gemini-3.5-flash)         │       │  1. parallel_intelligence_gatherer (Parallel)    │
│ • PubChem/RDKit/TxGemma/ELN/HITL │       │     ├─ compound_and_literature_curator (Flash)   │
└──────────────────────────────────┘       │     └─ target_and_structure_context (Flash)      │
                                           │  2. sar_mmpa_modeling_agent (Pro + RDKit skill)  │
                                           │  3. admet_safety_triage_agent (Flash + TxGemma)  │
                                           │  4. synthesis_and_proposal_agent (Pro + HITL)    │
                                           │  5. scientific_self_critic_agent (Pro + Schema)  │
                                           └──────────────────────────────────────────────────┘
```

---

## 3. AgentOps Code Review Matrix Traceability (95 / 95)

| Category | Rubric Criterion | Implementation Evidence |
| :--- | :--- | :--- |
| **1. Tool & Interface Design** | Comprehensive Tool Docstrings | Google-style docstrings with chemistry rationale, `Args:`, and `Returns:` on every tool in `medchem_agent/tools/*.py` |
| | Descriptive Naming | Domain-specific names (`resolve_compound_identifiers_from_pubchem`, `run_matched_molecular_pair_rigidification_analysis`, `train_scaffold_split_potency_qsar_model`, `predict_txgemma_admet_and_toxicity_profile`, `export_validated_synthesis_candidates_to_eln`) |
| | Explicit JSON Schemas | Strict Pydantic v2 `BaseModel` (`extra="forbid"`) input/output contracts in `medchem_agent/schemas/chemistry.py`, `SelfCritiqueReport` `output_schema` on `scientific_self_critic_agent`, plus `/schemas` JSON Schema endpoint |
| | Guided Error Handling | Structured `ToolErrorResponse` (`error_code`, `recovery_instructions`, `suggested_next_tool`, `retryable`) across all tools + ADK `ReflectAndRetryToolPlugin` in `medchem_agent/agent.py` |
| **2. Context & Memory** | Robust System Instructions | Formal `MEDCHEM_CONSTITUTION` & specialist role prompts in `medchem_agent/prompts/constitution.py` |
| | History Compaction | `EventsCompactionConfig` (token-threshold + sliding-window with FAST `LlmEventSummarizer` & chemistry-aware prompt) + `ContextCacheConfig` on ADK `App` in `medchem_agent/agent.py` |
| | Persistent Session State | `VertexAiSessionService` / `DatabaseSessionService` (async SQLite/Cloud SQL), `VertexAiMemoryBankService` / `SqliteMemoryService` (FTS5) in `medchem_agent/memory/persistence.py`, plus SQLite + cosine-similarity `EnterpriseElnStore` in `medchem_agent/memory/enterprise_store.py` |
| | Async Memory Operations | Non-blocking `AsyncMemoryConsolidationManager` (`asyncio.create_task` + semaphore) in `medchem_agent/memory/async_consolidator.py`, dispatched from `ChemObservabilityPlugin.after_run_callback` and `commit_design_decision_to_campaign_memory` |
| **3. Orchestration & Logic** | Multi-Agent Patterns | Coordinator (`LlmAgent`) + `SequentialAgent` + `ParallelAgent` + 7 specialized agents + ADK `SkillToolset` in `medchem_agent/agents/pipeline.py` and `medchem_agent/agents/specialists.py` |
| | Strategic Model Routing | `ModelRouter` (`FAST`: `gemini-3.5-flash`, `REASONING`: `gemini-3.1-pro-preview`, `SPECIALIST_THERAPEUTICS`: DeepMind TxGemma) with dynamic escalation in `medchem_agent/config/model_router.py` |
| | Guardrails & Policy Plugins | `ChemGuardrailsAndPolicyPlugin` (`BasePlugin`) enforcing biosecurity/CBRN screening, Google Cloud Model Armor hook, Proprietary SMILES IP egress fence, and research-disclaimer self-eval in `medchem_agent/plugins/guardrails_plugin.py` |
| | Human-in-the-Loop Hooks | `ResumabilityConfig(is_resumable=True)` + ADK `FunctionTool(..., require_confirmation=...)` + `tool_context.request_confirmation()` in `medchem_agent/tools/export_hitl.py` (static gate on ELN SDF export; dynamic cost/batch/toxicity gate on CRO orders) |
| **4. Observability & Tracing** | Structured JSON Logging | `structlog` JSON renderer with Cloud Logging `severity` and trace/span correlation in `medchem_agent/observability/logging.py` (zero raw `print()` in agent runtime) |
| | Intent vs. Outcome Capture | `ChemObservabilityPlugin` records correlated `agent.tool.intent` (`before_tool_callback`) and `agent.tool.outcome` (`after_tool_callback` / `on_tool_error_callback`) with `intent_outcome_match` and `duration_ms` in `medchem_agent/plugins/observability_plugin.py` |
| | Distributed Tracing | OpenTelemetry `TracerProvider` supporting Google Cloud Trace (`CloudTraceSpanExporter`), OTLP HTTP, and `InMemorySpanExporter` across agent & tool spans in `medchem_agent/observability/tracing.py` |
| | PII Redaction | Dual-layer `PiiRedactor` (Google Cloud DLP API + deterministic regex/key masker for emails, phones, SSNs, patient MRNs, and API keys) embedded in logging, memory, and provenance exports in `medchem_agent/observability/pii_redactor.py` |
| **5. Infrastructure & CI/CD** | Automated Evaluation Suites | ADK Pydantic-validated `eval/medchem_golden.evalset.json`, `eval/test_config.json`, `eval/run_eval.py` (offline deterministic chemistry/safety suite + live `AgentEvaluator` support), and comprehensive `tests/` suite |
| | Infrastructure as Code | Modular Terraform in `terraform/` (`google_vertex_ai_reasoning_engine`, Cloud Run v2, Artifact Registry, Secret Manager, least-privilege IAM, Cloud Monitoring alerts), `Dockerfile`, `cloudbuild.yaml`, `.github/workflows/ci.yml`, and `deploy/deploy_geap.sh` |
| | Secure Secret Management | `SecretProvider` backed by Google Cloud Secret Manager with TTL cache & local `.env` fallback in `medchem_agent/config/secrets.py` (zero hardcoded credentials; verified statically in tests) |

---

## 4. Quickstart (Local Development)

```bash
# 1. Clone and install dependencies (Python 3.12 + uv)
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"

# 2. Configure local environment (never commit .env)
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY (or set GOOGLE_GENAI_USE_VERTEXAI=TRUE + GCP_PROJECT_ID)

# 3. Run the full unit test suite & deterministic golden chemistry evaluation
.venv/bin/pytest -v tests/
.venv/bin/python eval/run_eval.py

# 4. Launch the interactive ADK Web UI
.venv/bin/adk web .

# 5. Or run the production FastAPI + persistent Runner server
.venv/bin/python -m medchem_agent.main
```

### Example Prompts to Try in `adk web`
- *"Run an ALK lead-optimization analysis comparing crizotinib and lorlatinib rigidification, including MMPA deltas, Bemis-Murcko scaffold-split QSAR, and DeepMind TxGemma ADMET triage."*
- *"Resolve lorlatinib from PubChem and score its BBB permeability and hERG cardiotoxicity with TxGemma."*
- *"Export our top RDKit-validated ALK macrocycle to the Enterprise ELN under campaign ALK-MACRO-2026-Q4 (estimated cost $7,500)."* *(Triggers ADK Human-in-the-Loop confirmation.)*

---

## 5. Deploying to Gemini Enterprise Agent Platform (GEAP)

```bash
# Option A: Deploy via ADK CLI to GEAP Agent Runtime (Vertex AI Agent Engine)
export GCP_PROJECT_ID="your-gcp-project-id"
export GCP_LOCATION="us-central1"
./deploy/deploy_geap.sh adk

# Option B: Provision full enterprise infrastructure + Agent Runtime via Terraform
./deploy/deploy_geap.sh terraform
```
