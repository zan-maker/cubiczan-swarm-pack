"""
Consensus Engine — Adversarial debate with LMSR scoring.
Solves the sycophancy problem via heterogeneous models, contrarian roles,
and independent parallel initialization.

Based on CONSENSAGENT (ACL 2025) and Google Research scaling principles.
"""

import json
import math
import os
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class AgentVote:
    agent_id: str
    model_name: str
    position: str  # The agent's analysis/position
    confidence: float  # 0.0-1.0
    reasoning: str
    is_contrarian: bool = False


@dataclass
class ConsensusResult:
    final_position: str
    consensus_score: float  # 0.0-1.0 via LMSR
    votes: list[AgentVote] = field(default_factory=list)
    debate_rounds: int = 0
    escalate_to_human: bool = False
    dissenting_views: list[str] = field(default_factory=list)


class LMSRScorer:
    """Logarithmic Market Scoring Rule for consensus measurement."""

    def __init__(self, liquidity: float = 100.0):
        self.b = liquidity

    def cost(self, quantities: list[float]) -> float:
        return self.b * math.log(sum(math.exp(q / self.b) for q in quantities))

    def price(self, quantities: list[float], outcome_idx: int) -> float:
        exp_vals = [math.exp(q / self.b) for q in quantities]
        return exp_vals[outcome_idx] / sum(exp_vals)

    def score_consensus(self, confidences: list[float]) -> float:
        """Score how much agreement exists. 1.0 = full consensus, 0.0 = split."""
        if not confidences:
            return 0.0
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        agreement = 1.0 - min(variance * 4, 1.0)  # Scale variance to [0,1]
        return round(agreement * mean_conf, 3)


DEBATE_SYSTEM = """You are an expert analyst in a swarm intelligence system.
You must provide your INDEPENDENT analysis. Do NOT agree with other agents
unless their reasoning genuinely convinces you. You are rewarded for
intellectual honesty, not for consensus.

If you are assigned the CONTRARIAN role, you MUST:
1. Challenge the majority position with logical counter-arguments
2. Identify blind spots, unstated assumptions, and edge cases
3. Provide a proof or evidence for your contrarian stance
4. Only concede if presented with irrefutable evidence"""

CONTRARIAN_ADDENDUM = """
YOUR ROLE: CONTRARIAN AGENT
You MUST argue AGAINST the emerging consensus. Find flaws, edge cases,
and alternative explanations. Your job is stress-testing, not agreement.
Provide specific logical proofs for your contrarian position."""


class ConsensusEngine:
    def __init__(self):
        self.models = self._load_models()
        self.scorer = LMSRScorer()
        self.threshold = float(os.getenv("CONSENSUS_THRESHOLD", "0.75"))
        self.max_rounds = 3

    def _load_models(self) -> list[dict]:
        """Load heterogeneous model pool from env."""
        models = []
        for i in range(1, 6):
            key = os.getenv(f"AGENT_MODEL_{i}_API_KEY")
            url = os.getenv(f"AGENT_MODEL_{i}_BASE_URL")
            name = os.getenv(f"AGENT_MODEL_{i}_NAME")
            if key and url and name:
                models.append({
                    "client": OpenAI(api_key=key, base_url=url),
                    "model": name,
                    "id": f"agent-{i}",
                })
        # Fallback to primary model if none configured
        if not models:
            models.append({
                "client": OpenAI(
                    api_key=os.getenv("LLM_API_KEY"),
                    base_url=os.getenv("LLM_BASE_URL"),
                ),
                "model": os.getenv("LLM_MODEL_NAME", "qwen2.5:32b"),
                "id": "agent-primary",
            })
        return models

    def _query_agent(
        self, model_cfg: dict, task: str, context: str, is_contrarian: bool
    ) -> AgentVote:
        system = DEBATE_SYSTEM
        if is_contrarian:
            system += CONTRARIAN_ADDENDUM

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"TASK: {task}\n\n"
                f"CONTEXT FROM OTHER AGENTS:\n{context}\n\n"
                "Provide your analysis in JSON:\n"
                '{"position": "your analysis", '
                '"confidence": 0.0-1.0, '
                '"reasoning": "key reasoning"}'
            )},
        ]

        response = model_cfg["client"].chat.completions.create(
            model=model_cfg["model"],
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return AgentVote(
            agent_id=model_cfg["id"],
            model_name=model_cfg["model"],
            position=result["position"],
            confidence=float(result["confidence"]),
            reasoning=result["reasoning"],
            is_contrarian=is_contrarian,
        )

    def run_consensus(self, task: str, cluster_size: int = 4) -> ConsensusResult:
        """Run adversarial consensus with heterogeneous agents."""
        agents = self.models[:cluster_size]
        if not agents:
            raise ValueError("No agent models configured")

        # Assign contrarian role (last agent in cluster)
        contrarian_idx = len(agents) - 1

        all_votes: list[AgentVote] = []
        context = "No prior analysis available. You are the first to analyze."

        for round_num in range(self.max_rounds):
            round_votes = []
            for i, agent in enumerate(agents):
                is_contrarian = (i == contrarian_idx)
                vote = self._query_agent(agent, task, context, is_contrarian)
                round_votes.append(vote)

            all_votes.extend(round_votes)

            # Calculate consensus
            confidences = [v.confidence for v in round_votes if not v.is_contrarian]
            consensus_score = self.scorer.score_consensus(confidences)

            if consensus_score >= self.threshold:
                # Consensus reached
                majority = max(round_votes, key=lambda v: v.confidence)
                dissents = [
                    v.position for v in round_votes
                    if v.is_contrarian and v.confidence > 0.5
                ]
                return ConsensusResult(
                    final_position=majority.position,
                    consensus_score=consensus_score,
                    votes=all_votes,
                    debate_rounds=round_num + 1,
                    escalate_to_human=False,
                    dissenting_views=dissents,
                )

            # Build context for next round
            context = "\n\n".join(
                f"[{v.agent_id} ({v.model_name})] "
                f"{'[CONTRARIAN] ' if v.is_contrarian else ''}"
                f"Position: {v.position}\n"
                f"Confidence: {v.confidence}\n"
                f"Reasoning: {v.reasoning}"
                for v in round_votes
            )

        # No consensus — escalate to human
        majority = max(all_votes, key=lambda v: v.confidence)
        return ConsensusResult(
            final_position=majority.position,
            consensus_score=consensus_score,
            votes=all_votes,
            debate_rounds=self.max_rounds,
            escalate_to_human=True,
            dissenting_views=[
                v.position for v in all_votes if v.is_contrarian
            ],
        )
