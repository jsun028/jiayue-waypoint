from typing import Iterable
import inspect
from registry import GLOBAL_UDF_REGISTRY

def _udf_name(candidate: object) -> str:
    """Best-effort extraction of a predicate name from user-provided input."""
    if isinstance(candidate, str):
        return candidate
    return getattr(candidate, "__name__", "<anonymous>")

def _resolve_udf_callable(name: str, candidate: object):
    """Locate the callable implementing a predicate, if available."""
    registry_funcs = GLOBAL_UDF_REGISTRY.get_all_udfs()
    func = None

    if callable(candidate):
        func = candidate
    else:
        func = registry_funcs.get(name)

    return func


def _format_udf_info(name: str, candidate: object) -> str:
    """Return docstring-based summary for prompt conditioning."""
    func = _resolve_udf_callable(name, candidate)

    if func is None:
        return name

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        signature = "(…)"

    doc_raw = func.__doc__ or ""
    doc_lines = [line.rstrip() for line in doc_raw.strip().splitlines() if line.strip()]
    summary = f"{name}{signature}"
    if doc_lines:
        summary += "\n    " + "\n    ".join(doc_lines)
    return summary


def _format_available_udfs_for_prompt(available_udfs: Iterable[object]) -> list[str]:
    """Normalize UDF inputs to unique, alphabetized signature strings."""
    seen = set()
    formatted = []
    for item in available_udfs:
        name = _udf_name(item)
        if name in seen:
            continue
        seen.add(name)
        formatted.append(_format_udf_info(name, item))
    return sorted(formatted)

# usage example
# available = _format_available_udfs_for_prompt(
#     GLOBAL_UDF_REGISTRY.get_all_udfs().keys()
# )

# for doc in available:
#     print(doc)

AVAILABLE_UDFS = ", ".join(GLOBAL_UDF_REGISTRY.get_all_udfs().keys())
print(AVAILABLE_UDFS)