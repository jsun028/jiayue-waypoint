import os
from .base import BaseProvider

# Google AI for Developers – Gemini API (python)
# Docs: https://ai.google.dev/api/generate-content  + quickstart
import google.generativeai as genai

class GeminiProvider(BaseProvider):
    def __init__(self, model: str = "gemini-1.5-pro"):
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, *, temperature: float, max_output_tokens: int, timeout_s: int) -> str:
        resp = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
        return resp.text or ""
