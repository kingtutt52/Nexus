"""
Claude API client with prompt caching, streaming, and citation support.
Optimized for high-frequency enterprise agent calls.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import anthropic

MODEL = "claude-opus-4-7"
CACHE_CONTROL = {"type": "ephemeral"}


@dataclass
class AgentMessage:
    role: str
    content: str
    agent_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class APIStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    total_calls: int = 0
    latency_ms: list[float] = field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_creation_tokens + self.cache_read_tokens
        return self.cache_read_tokens / total if total > 0 else 0.0

    @property
    def effective_cost_reduction(self) -> float:
        """Cache reads cost 10% of normal input tokens."""
        if self.input_tokens == 0:
            return 0.0
        saved = self.cache_read_tokens * 0.9
        return saved / (self.input_tokens + self.cache_read_tokens)


class NexusLLMClient:
    """
    Wrapper around Anthropic API optimized for multi-agent enterprise workloads.

    Key optimizations:
    - Prompt caching on system prompts + large context (domain knowledge, KPI schemas)
    - Streaming for long-running analysis tasks
    - Automatic retry with exponential backoff
    - Token tracking for cost visibility
    """

    def __init__(self, api_key: Optional[str] = None, model: str = MODEL):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.stats = APIStats()

    def chat(
        self,
        messages: list[dict],
        system: Optional[str | list[dict]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        cache_system: bool = True,
    ) -> str:
        """
        Single-turn chat with optional system prompt caching.
        Returns assistant content string.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system:
            if isinstance(system, str) and cache_system:
                kwargs["system"] = [
                    {"type": "text", "text": system, "cache_control": CACHE_CONTROL}
                ]
            elif isinstance(system, list):
                kwargs["system"] = system
            else:
                kwargs["system"] = system

        t0 = time.perf_counter()
        response = self._call_with_retry(kwargs)
        self.stats.latency_ms.append((time.perf_counter() - t0) * 1000)

        self._update_stats(response.usage)
        return response.content[0].text

    def chat_with_citations(
        self,
        messages: list[dict],
        documents: list[dict],
        system: Optional[str] = None,
    ) -> tuple[str, list[dict]]:
        """
        Chat with document citations enabled.
        documents: [{"title": "...", "content": "...", "source": "..."}]
        Returns (response_text, citations_list)
        """
        doc_blocks = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": d["content"]},
                "title": d.get("title", ""),
                "citations": {"enabled": True},
                "cache_control": CACHE_CONTROL,
            }
            for d in documents
        ]

        enriched_messages = messages.copy()
        if enriched_messages and enriched_messages[-1]["role"] == "user":
            enriched_messages[-1] = {
                "role": "user",
                "content": doc_blocks + [{"type": "text", "text": enriched_messages[-1]["content"]}],
            }

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": enriched_messages,
        }
        if system:
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": CACHE_CONTROL}]

        response = self._call_with_retry(kwargs)
        self._update_stats(response.usage)

        text_parts = []
        citations = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            if hasattr(block, "citations"):
                citations.extend(block.citations)

        return "".join(text_parts), citations

    def debate_round(
        self,
        topic: str,
        agent_a_position: str,
        agent_b_position: str,
        context: str,
        rounds: int = 2,
    ) -> list[AgentMessage]:
        """
        Run structured debate between two agent positions.
        Used by the orchestrator for hypothesis validation.
        """
        transcript: list[AgentMessage] = []
        system_base = f"""You are a rigorous analytical agent debating enterprise AI interventions.
Context: {context}

You must:
1. Ground every claim in quantitative evidence from the context
2. Explicitly acknowledge uncertainty and confidence levels
3. Identify failure modes and edge cases
4. Conclude with a probability estimate (0-100%) for your position"""

        # Agent A opens
        a_msg = self.chat(
            messages=[{"role": "user", "content": f"Topic: {topic}\n\nYour position: {agent_a_position}\n\nPresent your opening argument."}],
            system=system_base,
            cache_system=True,
        )
        transcript.append(AgentMessage("assistant", a_msg, "Advocate"))

        # Agent B responds
        b_msg = self.chat(
            messages=[
                {"role": "user", "content": f"Topic: {topic}\n\nYour position: {agent_b_position}\n\nOpponent's argument:\n{a_msg}\n\nRespond to their argument and present your own."},
            ],
            system=system_base,
            cache_system=True,
        )
        transcript.append(AgentMessage("assistant", b_msg, "Adversary"))

        # Rebuttals
        for _ in range(rounds - 1):
            a_rebuttal = self.chat(
                messages=[{"role": "user", "content": f"Counter-argument: {b_msg}\n\nProvide your rebuttal, updating your probability estimate."}],
                system=system_base,
                cache_system=True,
            )
            transcript.append(AgentMessage("assistant", a_rebuttal, "Advocate"))
            b_msg = self.chat(
                messages=[{"role": "user", "content": f"Counter-argument: {a_rebuttal}\n\nFinal rebuttal."}],
                system=system_base,
                cache_system=True,
            )
            transcript.append(AgentMessage("assistant", b_msg, "Adversary"))

        return transcript

    def _call_with_retry(self, kwargs: dict, max_retries: int = 3) -> Any:
        for attempt in range(max_retries):
            try:
                return self.client.messages.create(**kwargs)
            except anthropic.RateLimitError:
                time.sleep(2 ** attempt)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError(f"API call failed after {max_retries} retries")

    def _update_stats(self, usage: Any) -> None:
        self.stats.total_calls += 1
        self.stats.input_tokens += getattr(usage, "input_tokens", 0)
        self.stats.output_tokens += getattr(usage, "output_tokens", 0)
        self.stats.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0)
        self.stats.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0)
