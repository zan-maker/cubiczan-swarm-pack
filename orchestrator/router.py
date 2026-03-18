"""
MoE Router — Lightweight classifier that routes incoming tasks to specialist clusters.
Uses nano models ($0.02-0.20/M tokens) to replace brute-force all-agent processing.
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI


class Domain(str, Enum):
    FINANCIAL = "financial"
    CYBERSECURITY = "cybersecurity"
    BUSINESS_INTEL = "business-intel"
    PREDICTIVE_SIM = "predictive-sim"
    CONTENT_MARKETING = "content-marketing"
    HEALTHCARE = "healthcare"
    POLITICAL = "political"
    REAL_ESTATE = "real-estate"
    TALENT_HR = "talent-hr"


@dataclass
class RoutingDecision:
    domain: Domain
    confidence: float
    reasoning: str
    complexity: str  # "low", "medium", "high"
    requires_consensus: bool


ROUTING_PROMPT = """You are a task routing classifier for the Cubiczan Agent Swarm.
Classify the incoming task into exactly ONE primary domain and assess its complexity.

Available domains:
- financial: Market analysis, trading signals, portfolio construction, M&A screening, commodities, crypto
- cybersecurity: SOC triage, threat hunting, vulnerability scanning, dark web monitoring, compliance
- business-intel: Competitive analysis, M&A targets, market entry, pricing strategy, supply chain
- predictive-sim: Social dynamics forecasting, scenario modeling, prediction markets, simulation
- content-marketing: Content creation, ad optimization, SEO, influencer selection, trend detection
- healthcare: Drug discovery, clinical trials, regulatory navigation, biomarker analysis
- political: Election prediction, policy impact, geopolitical risk, social movement forecasting
- real-estate: Property valuation, gentrification prediction, site selection, zoning analysis
- talent-hr: Hiring optimization, retention signals, compensation tracking, team composition

Respond in JSON:
{
  "domain": "<domain-key>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence>",
  "complexity": "low|medium|high",
  "requires_consensus": <true if high-stakes or multi-domain>
}"""


class MoERouter:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("ROUTER_MODEL_API_KEY", os.getenv("LLM_API_KEY")),
            base_url=os.getenv("ROUTER_MODEL_BASE_URL", os.getenv("LLM_BASE_URL")),
        )
        self.model = os.getenv("ROUTER_MODEL_NAME", "qwen2.5:7b")
        active = os.getenv("ACTIVE_DOMAINS", "financial,cybersecurity,business-intel")
        self.active_domains = {d.strip() for d in active.split(",")}

    def route(self, task_description: str) -> RoutingDecision:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ROUTING_PROMPT},
                {"role": "user", "content": task_description},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        domain = Domain(result["domain"])

        if domain.value not in self.active_domains:
            raise ValueError(
                f"Domain '{domain.value}' is not active. "
                f"Active: {self.active_domains}"
            )

        return RoutingDecision(
            domain=domain,
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            complexity=result.get("complexity", "medium"),
            requires_consensus=result.get("requires_consensus", False),
        )

    def route_multi(self, task_description: str) -> list[RoutingDecision]:
        """Route to multiple domains for cross-domain tasks."""
        primary = self.route(task_description)
        if not primary.requires_consensus:
            return [primary]

        # For cross-domain tasks, identify secondary domains
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "Given this task that requires multi-domain analysis, "
                    "list ALL relevant domains as a JSON array of domain keys. "
                    f"Available: {list(self.active_domains)}"
                )},
                {"role": "user", "content": task_description},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        domains = json.loads(response.choices[0].message.content).get("domains", [])
        decisions = [primary]
        for d in domains:
            if d != primary.domain.value and d in self.active_domains:
                decisions.append(RoutingDecision(
                    domain=Domain(d),
                    confidence=primary.confidence * 0.8,
                    reasoning=f"Secondary domain for: {primary.reasoning}",
                    complexity=primary.complexity,
                    requires_consensus=True,
                ))
        return decisions
