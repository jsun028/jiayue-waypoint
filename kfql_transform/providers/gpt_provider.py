import os
from .base import BaseProvider

# Official SDK & Responses API (preferred).
# Docs: OpenAI Python SDK + Responses API. 
# See: https://github.com/openai/openai-python  and
#      https://openai.com/index/introducing-gpt-5-for-developers/
from openai import OpenAI

class OpenAIProvider(BaseProvider):
    def __init__(self, model: str = "gpt-5", reasoning_effort: str = "high", verbosity: str = "low"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

    def generate(self, prompt: str, *, temperature: float, max_output_tokens: int, timeout_s: int) -> str:
        # Responses API (single-turn text generation)
        # Cookbook & blog mention new params like reasoning_effort, verbosity.
        # Refs: openai-python README + GPT-5 dev post.

        kwargs = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }

        # gpt-5 reasoning model → temperature not supported
        if not self.model.startswith("gpt-5"):
            kwargs["temperature"] = temperature

        # reasoning effort is only supported by gpt-5
        if self.model.startswith("gpt-5"):
            kwargs["reasoning"] = {"effort": self.reasoning_effort}

        response = self.client.responses.create(**kwargs)
        return getattr(response, "output_text", None) or response.output[0].content[0].text