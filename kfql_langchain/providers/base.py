from abc import ABC, abstractmethod

class BaseProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, temperature: float, max_output_tokens: int, timeout_s: int) -> str:
        """Return raw model text output."""
        ...
