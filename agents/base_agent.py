"""
Base Heterogeneous Agent — Foundation for all specialist agents.
Supports multiple LLM backends for anti-sycophancy diversity.
"""

import os
import json
import uuid
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class AgentProfile:
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: str = "analyst"
    domain: str = "general"
    model_name: str = "qwen2.5:32b"
    weight: float = 0.25
    is_contrarian: bool = False
    persona: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class AgentResponse:
    agent_id: str
    model_name: str
    role: str
    position: str
    confidence: float
    reasoning: str
    evidence: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


class BaseAgent:
    """Heterogeneous agent that can use any OpenAI-compatible LLM backend."""

    def __init__(self, profile: AgentProfile, client: OpenAI | None = None):
        self.profile = profile
        self.client = client or OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
        self.conversation_history: list[dict] = []

    def _build_system_prompt(self, domain_prompt: str = "") -> str:
        base = (
            f"You are Agent {self.profile.agent_id}, a {self.profile.role} "
            f"specialist in the {self.profile.domain} domain.\n\n"
        )
        if self.profile.persona:
            base += f"PERSONA: {self.profile.persona}\n\n"
        if domain_prompt:
            base += f"DOMAIN CONTEXT:\n{domain_prompt}\n\n"
        if self.profile.is_contrarian:
            base += (
                "YOUR ROLE: CONTRARIAN\n"
                "You MUST challenge the majority position with:\n"
                "1. Specific logical flaws in the consensus\n"
                "2. Counter-evidence or alternative explanations\n"
                "3. Edge cases the majority has overlooked\n"
                "You may ONLY concede if given irrefutable evidence.\n\n"
            )
        base += (
            "OUTPUT FORMAT (JSON):\n"
            '{"position": "your analysis", '
            '"confidence": 0.0-1.0, '
            '"reasoning": "key reasoning", '
            '"evidence": ["source1", "source2"]}'
        )
        return base

    def analyze(
        self,
        task: str,
        context: str = "",
        domain_prompt: str = "",
    ) -> AgentResponse:
        system = self._build_system_prompt(domain_prompt)
        user_msg = f"TASK: {task}"
        if context:
            user_msg += f"\n\nCONTEXT FROM OTHER AGENTS:\n{context}"

        messages = [
            {"role": "system", "content": system},
            *self.conversation_history,
            {"role": "user", "content": user_msg},
        ]

        response = self.client.chat.completions.create(
            model=self.profile.model_name,
            messages=messages,
            temperature=self.profile.temperature,
            max_tokens=self.profile.max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        # Strip <think> tags from reasoning models like DeepSeek-R1
        if "<think>" in content:
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        result = json.loads(content)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_msg})
        self.conversation_history.append({"role": "assistant", "content": content})

        return AgentResponse(
            agent_id=self.profile.agent_id,
            model_name=self.profile.model_name,
            role=self.profile.role,
            position=result.get("position", ""),
            confidence=float(result.get("confidence", 0.5)),
            reasoning=result.get("reasoning", ""),
            evidence=result.get("evidence", []),
        )

    def reset(self):
        """Clear conversation history for fresh analysis."""
        self.conversation_history = []


def create_cluster(
    domain: str,
    swarm_config: dict,
    models: list[dict],
) -> list[BaseAgent]:
    """Create a heterogeneous agent cluster from domain config."""
    composition = swarm_config.get("swarm_composition", [])
    agents = []

    for i, role_config in enumerate(composition):
        model_cfg = models[i % len(models)]
        is_contrarian = role_config["role"].lower() == "contrarian"
        profile = AgentProfile(
            role=role_config["role"],
            domain=domain,
            model_name=model_cfg["model"],
            weight=role_config["weight"],
            is_contrarian=is_contrarian,
            persona=role_config.get("description", ""),
        )
        client = OpenAI(
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
        )
        agents.append(BaseAgent(profile=profile, client=client))

    return agents
