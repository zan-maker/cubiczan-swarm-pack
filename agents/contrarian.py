"""
Contrarian Agent — Explicit adversarial role for consensus stress-testing.

Implements the anti-sycophancy measures from:
- CONSENSAGENT (ACL 2025): Same-model agents converge in 1-2 rounds
- Kimi K2.5 Agent Swarm: Heterogeneous subagent instantiation
- Google Research: Centralized hub-and-spoke contains errors to 4.4x
"""

import json
import os
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class ContrarianChallenge:
    claim_challenged: str
    logical_flaw: str
    counter_evidence: list[str]
    alternative_explanation: str
    probability_majority_wrong: float  # 0-100%
    concession_conditions: str


CONTRARIAN_SYSTEM = """You are a CONTRARIAN AGENT in the Cubiczan Agent Swarm.
Your role is adversarial stress-testing of the majority consensus.

You are NOT being difficult for its own sake. You are the last line of defense
against groupthink, confirmation bias, and sycophantic convergence.

MANDATORY OUTPUT STRUCTURE:
{
  "claim_challenged": "The specific majority claim you are challenging",
  "logical_flaw": "The precise logical flaw, unstated assumption, or gap in reasoning",
  "counter_evidence": ["Specific counter-evidence point 1", "Point 2"],
  "alternative_explanation": "A plausible alternative interpretation of the same data",
  "probability_majority_wrong": 0-100,
  "concession_conditions": "What evidence would make you concede (be specific)"
}

RULES:
1. Never agree with the majority without finding at least ONE flaw first
2. Challenge assumptions, not just conclusions
3. Consider: base rate neglect, survivorship bias, anchoring, availability heuristic
4. If the majority cites statistics, question the methodology and sample
5. If the majority cites experts, look for dissenting expert opinions
6. You may concede ONLY if irrefutable evidence addresses ALL your objections
7. Assign calibrated probability (not just "maybe wrong")"""


class ContrarianAgent:
    def __init__(self, model_name: str | None = None):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
        self.model = model_name or os.getenv("LLM_MODEL_NAME", "qwen2.5:32b")

    def challenge(
        self, majority_position: str, task_context: str
    ) -> ContrarianChallenge:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CONTRARIAN_SYSTEM},
                {"role": "user", "content": (
                    f"TASK CONTEXT:\n{task_context}\n\n"
                    f"MAJORITY CONSENSUS POSITION:\n{majority_position}\n\n"
                    "Challenge this position. Find the flaws."
                )},
            ],
            temperature=0.8,  # Higher temp for creative counter-arguments
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return ContrarianChallenge(
            claim_challenged=result.get("claim_challenged", ""),
            logical_flaw=result.get("logical_flaw", ""),
            counter_evidence=result.get("counter_evidence", []),
            alternative_explanation=result.get("alternative_explanation", ""),
            probability_majority_wrong=float(
                result.get("probability_majority_wrong", 50)
            ),
            concession_conditions=result.get("concession_conditions", ""),
        )

    def evaluate_concession(
        self,
        challenge: ContrarianChallenge,
        majority_rebuttal: str,
    ) -> dict:
        """Evaluate whether the majority's rebuttal warrants concession."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You previously challenged the majority position. "
                    "Evaluate their rebuttal against your concession conditions. "
                    "Be intellectually honest but rigorous.\n\n"
                    "Respond in JSON:\n"
                    '{"concede": true/false, '
                    '"reasoning": "why you concede or don\'t", '
                    '"remaining_objections": ["any unaddressed points"], '
                    '"updated_probability_wrong": 0-100}'
                )},
                {"role": "user", "content": (
                    f"YOUR CHALLENGE:\n"
                    f"Flaw: {challenge.logical_flaw}\n"
                    f"Counter-evidence: {challenge.counter_evidence}\n"
                    f"Concession conditions: {challenge.concession_conditions}\n\n"
                    f"MAJORITY REBUTTAL:\n{majority_rebuttal}"
                )},
            ],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
