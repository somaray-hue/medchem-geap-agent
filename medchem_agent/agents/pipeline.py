"""Multi-Agent Orchestration Patterns: Coordinator -> Sequential -> Parallel ADK Architecture.

Pattern topology:
  medchem_coordinator_agent (LlmAgent, REASONING tier + PreloadMemoryTool)
   ├── quick_chemistry_lookup_agent (LlmAgent, FAST tier — single-step queries)
   └── lead_optimization_pipeline (SequentialAgent — full drug-discovery design loop)
        ├── 1. parallel_intelligence_gatherer (ParallelAgent)
        │    ├── compound_and_literature_curator_agent (FAST tier + PubChem/ChEMBL skills)
        │    └── target_and_structure_context_agent (FAST tier + AlphaFold skill)
        ├── 2. sar_mmpa_modeling_agent (REASONING tier + RDKit MMPA/QSAR skill)
        ├── 3. admet_safety_triage_agent (FAST tier + DeepMind TxGemma endpoint)
        ├── 4. synthesis_and_proposal_agent (REASONING tier + HITL ELN export & CRO tools)
        └── 5. scientific_self_critic_agent (REASONING tier + SelfCritiqueReport schema)
"""

from __future__ import annotations

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

from medchem_agent.agents.specialists import (
    build_admet_triage_agent,
    build_curator_agent,
    build_quick_chemistry_lookup_agent,
    build_sar_mmpa_agent,
    build_self_critic_agent,
    build_synthesis_design_agent,
    build_target_context_agent,
)
from medchem_agent.config.model_router import ModelRouter, TaskComplexity
from medchem_agent.prompts.constitution import COORDINATOR_INSTRUCTION
from medchem_agent.tools.export_hitl import commit_design_decision_to_campaign_memory


def build_lead_optimization_pipeline(router: ModelRouter | None = None) -> SequentialAgent:
    """Assemble the 5-stage Sequential + Parallel medicinal chemistry lead-optimization workflow."""
    mr = router or ModelRouter()

    parallel_gatherer = ParallelAgent(
        name="parallel_intelligence_gatherer",
        description=(
            "Concurrently gathers ChEMBL/PubChem/EuropePMC chemical series and enterprise "
            "Target Product Profile / AlphaFold structural context."
        ),
        sub_agents=[
            build_curator_agent(mr),
            build_target_context_agent(mr),
        ],
    )

    return SequentialAgent(
        name="lead_optimization_pipeline",
        description=(
            "End-to-end medicinal chemistry lead-optimization pipeline: parallel data & target "
            "curation -> RDKit MMPA & scaffold-split QSAR -> DeepMind TxGemma ADMET triage -> "
            "RDKit-verified macrocycle design & HITL export -> scientific self-critique."
        ),
        sub_agents=[
            parallel_gatherer,
            build_sar_mmpa_agent(mr),
            build_admet_triage_agent(mr),
            build_synthesis_design_agent(mr),
            build_self_critic_agent(mr),
        ],
    )


def build_medchem_coordinator_agent(router: ModelRouter | None = None) -> LlmAgent:
    """Assemble the root Coordinator Agent with preloaded long-term memory and sub-agent routing."""
    mr = router or ModelRouter()
    route = mr.route(TaskComplexity.PLANNING)

    return LlmAgent(
        name="medchem_coordinator_agent",
        model=route.model_id,
        description=(
            "Enterprise Life Sciences Medicinal Chemistry Coordinator powered by Google ADK, "
            "GEAP, and DeepMind Agentic Cheminformatics + TxGemma."
        ),
        instruction=COORDINATOR_INSTRUCTION,
        tools=[
            PreloadMemoryTool(),
            commit_design_decision_to_campaign_memory,
        ],
        sub_agents=[
            build_quick_chemistry_lookup_agent(mr),
            build_lead_optimization_pipeline(mr),
        ],
    )
