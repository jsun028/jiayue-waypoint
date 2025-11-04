"""Minimal stub of the loguru API used within the project.

This avoids pulling the external dependency in constrained environments.
Only the small subset of methods that the codebase calls are implemented.
"""

from __future__ import annotations

from typing import Any


class _Logger:
    def add(self, *args: Any, **kwargs: Any) -> None:
        pass

    def info(self, *args: Any, **kwargs: Any) -> None:
        pass

    def warning(self, *args: Any, **kwargs: Any) -> None:
        pass

    def error(self, *args: Any, **kwargs: Any) -> None:
        pass

    def debug(self, *args: Any, **kwargs: Any) -> None:
        pass


logger = _Logger()

__all__ = ["logger"]
