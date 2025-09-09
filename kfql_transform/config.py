from dataclasses import dataclass
from typing import Optional, Literal

ProviderName = Literal["openai", "gemini"]

@dataclass
class LLMConfig:
    provider: ProviderName = "openai"
    ## openai
    openai_model: str = "gpt-5"           # or "gpt-5-mini"
    openai_reasoning_effort: str = "high" # "minimal" | "low" | "medium" | "high"
    openai_verbosity: str = "low"         # "low" | "medium" | "high"
    ## gemini
    gemini_model: str = "gemini-1.5-pro"  # e.g. "gemini-2.5-flash", "gemini-1.5-pro"
    ## common
    temperature: float = 0.2
    max_output_tokens: int = 4000
    timeout_s: int = 90

@dataclass
class TransformConfig:
    # knobs for the prompt
    command: str = "Transform the JSON annotations into a KeyframeQL query."
    include_query_dsl: bool = True
    include_examples: bool = True
    require_code_fence: bool = True
    # your preferred default velocity threshold for “driving fast”
    default_fast_velocity: float = 2.0
