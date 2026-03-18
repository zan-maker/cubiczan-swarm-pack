"""
Anti-Sycophancy Module — Prevents false consensus in multi-agent systems.

Implements mitigations from CONSENSAGENT (ACL 2025) and MiroFish analysis:
1. Heterogeneous base models (3-5 different LLMs per cluster)
2. Explicit contrarian agent roles with logical proof requirements
3. Anti-sycophancy prompting with independent parallel initialization
4. PARL-inspired training: rewards true parallelism, penalizes fake consensus
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DiversityMetrics:
    model_diversity: float      # Unique models / total agents
    position_variance: float    # Variance in position embeddings
    cosine_similarity: float    # Avg pairwise cosine sim of responses
    sycophancy_risk: str        # "low", "medium", "high", "critical"


def compute_text_similarity(texts: list[str]) -> float:
    """Simple word-overlap similarity as a proxy for cosine similarity.
    In production, use sentence-transformers for proper embeddings.
    """
    if len(texts) < 2:
        return 0.0

    word_sets = [set(t.lower().split()) for t in texts]
    similarities = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            intersection = word_sets[i] & word_sets[j]
            union = word_sets[i] | word_sets[j]
            if union:
                similarities.append(len(intersection) / len(union))
    return sum(similarities) / len(similarities) if similarities else 0.0


def assess_sycophancy_risk(
    model_names: list[str],
    agent_positions: list[str],
    confidences: list[float],
) -> DiversityMetrics:
    """Assess the risk of sycophantic convergence in the current round."""

    # Model diversity: unique models / total agents
    unique_models = len(set(model_names))
    model_diversity = unique_models / len(model_names) if model_names else 0.0

    # Position variance via text similarity
    cosine_sim = compute_text_similarity(agent_positions)

    # Confidence variance
    conf_array = np.array(confidences) if confidences else np.array([0.0])
    position_variance = float(np.var(conf_array))

    # Risk assessment thresholds (from CONSENSAGENT paper)
    if cosine_sim > 0.95 and model_diversity < 0.5:
        risk = "critical"
    elif cosine_sim > 0.85:
        risk = "high"
    elif cosine_sim > 0.70:
        risk = "medium"
    else:
        risk = "low"

    return DiversityMetrics(
        model_diversity=round(model_diversity, 3),
        position_variance=round(position_variance, 4),
        cosine_similarity=round(cosine_sim, 4),
        sycophancy_risk=risk,
    )


# Anti-sycophancy prompt templates
INDEPENDENT_INIT_PROMPT = """CRITICAL INSTRUCTION: You are initializing your analysis INDEPENDENTLY.
You have NOT seen any other agent's work. Form your OWN position based
solely on the data provided. Do NOT hedge or qualify excessively.
State your genuine assessment with calibrated confidence."""

CONTRARIAN_PROOF_PROMPT = """You are the designated CONTRARIAN for this analysis round.
You MUST provide a structured counter-argument:

1. CLAIM: State the majority position you are challenging
2. FLAW: Identify the specific logical flaw or assumption
3. EVIDENCE: Provide counter-evidence or alternative explanation
4. PROBABILITY: Estimate the probability the majority is wrong (0-100%)

You may only CONCEDE if the majority provides irrefutable evidence
that addresses ALL of your objections. Partial addressing is insufficient."""

DIVERSITY_ENFORCEMENT_PROMPT = """DIVERSITY CHECK: The system has detected high similarity
between agent responses (cosine similarity > {sim:.2f}).
You MUST differentiate your analysis by:
1. Using a different analytical framework
2. Weighting different evidence
3. Considering a different time horizon
4. Applying a different risk model
Failure to differentiate indicates sycophantic convergence."""
