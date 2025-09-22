import os
from .base import BaseProvider

# Official SDK & Responses API (preferred).
# Docs: OpenAI Python SDK + Responses API. 
# See: https://github.com/openai/openai-python  and
#      https://openai.com/index/introducing-gpt-5-for-developers/
from openai import OpenAI

class OpenAIProvider(BaseProvider):
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model

    def generate(self, prompt: str, *, temperature: float, max_output_tokens: int, timeout_s: int) -> str:
        kwargs = {
            "model": self.model,
            "input": prompt,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }

        response = self.client.responses.create(**kwargs)

        # gpt-4o는 reasoning 모델이 아니므로 output_text로 안전하게 꺼낼 수 있음
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()

        # fallback: 구형 SDK나 응답 구조 다를 때
        if getattr(response, "output", None):
            try:
                return response.output[0].content[0].text
            except Exception:
                pass

        raise RuntimeError(f"No usable text found in response: {response}")