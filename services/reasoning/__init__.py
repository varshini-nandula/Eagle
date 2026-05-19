"""
services/reasoning — VLM captioning and LLM reasoning layer.

Set ``VLM_PROVIDER=mock`` to use the deterministic mock implementations
that require no GPU or external API.
"""
from services.reasoning.mock_vlm import (
    MockVLMCaptioner,
    MockLLMReasoner,
    ReasoningOutput,
    get_vlm_captioner,
    get_llm_reasoner,
)

__all__ = [
    "MockVLMCaptioner",
    "MockLLMReasoner",
    "ReasoningOutput",
    "get_vlm_captioner",
    "get_llm_reasoner",
]
