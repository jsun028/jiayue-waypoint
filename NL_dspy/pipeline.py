"""DSPy implementation of the NL → keyframe query specification pipeline."""

from __future__ import annotations

import inspect
import json
import pickle
from dataclasses import dataclass
from typing import Iterable, Optional

import dspy

from NL.registry import GLOBAL_UDF_REGISTRY
from NL.specs import QuerySpec, PredicateAtom, PredicateExpr


PROMPT_HEADER = """You translate traffic scene descriptions into JSON specs for the keyframe query engine.
- Define keyframes as salient frames that capture important transitions/events.
- Name keyframes k1, k2, k3, ... in temporal order.
- Output ONLY valid JSON; no markdown code fences or commentary.
- Use object aliases (car1, car2, pedestrian1, ...) unless NL input specifies otherwise.
- Use degrees for angles unless NL explicitly requests radians.
- Prefer velocity_above(2.0..5.0) for "moving" and velocity_below(2.0..3.0) for "stopped".
- For "right turn" events, add a trajectory constraint with template="right_arc".
- Allowed predicates (name and signature):{available_udfs}
- You may propose a new predicate only if absolutely required.
- Always include a concise "explanation" string that summarizes the reasoning behind the design of objects, keyframes, and constraints.
"""


FEWSHOT_USER = (
    "Right-turn scenario: car1 is initially moving opposite of car2, then after ~3s, "
    "they differ by ~90°. Car1 stops then accelerates into a right arc (~135°)."
)

FEWSHOT_JSON = {
    "objects": {
        "counts": {"car": 2},
        "aliases": {
            "car1": {"class": "car", "idx": 0},
            "car2": {"class": "car", "idx": 1},
        },
    },
    "keyframes": [
        {
            "name": "k1",
            "where": {
                "op": "AND",
                "args": [
                    {
                        "op": "ATOM",
                        "atom": {
                            "type": "heading_diff_to",
                            "obj": "car1",
                            "other_obj": "car2",
                            "value": 180.0,
                            "tol": 15.0,
                        },
                    },
                    {
                        "op": "ATOM",
                        "atom": {"type": "velocity_above", "obj": "car1", "value": 2.0},
                    },
                    {
                        "op": "ATOM",
                        "atom": {"type": "velocity_above", "obj": "car2", "value": 2.0},
                    },
                ],
            },
        },
        {
            "name": "k2",
            "where": {
                "op": "AND",
                "args": [
                    {
                        "op": "ATOM",
                        "atom": {
                            "type": "heading_diff_to",
                            "obj": "car1",
                            "other_obj": "car2",
                            "value": 90.0,
                            "tol": 15.0,
                        },
                    },
                    {
                        "op": "ATOM",
                        "atom": {"type": "velocity_above", "obj": "car1", "value": 2.0},
                    },
                    {
                        "op": "ATOM",
                        "atom": {"type": "velocity_above", "obj": "car2", "value": 2.0},
                    },
                ],
            },
        },
    ],
    "constraints": [
        {"kind": "always", "anchor": None, "target": "k1", "duration_sec": 3.0, "tol": 0.01},
        {"kind": "always", "anchor": None, "target": "k2", "duration_sec": 3.0, "tol": 0.01},
        {
            "kind": "interframe",
            "anchor": "k1",
            "target": "k2",
            "time_shift": 3.0,
            "comparators": [{"type": "heading_diff", "value": 90.0, "tol": 15.0}],
        },
        {
            "kind": "trajectory",
            "obj": "car1",
            "start": "k1",
            "end": "k2",
            "template": "right_arc",
            "angle_rad": 4.71238898038469,
            "deviation_strength": 0.0,
        },
    ],
    "explanation": (
        "car1 and car2 begin opposed while moving, then car1 slows to set up a right turn, "
        "and over ~3s completes a right-arc trajectory to align 90° from car2."
    ),
}


def _json_dumps(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


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


class SpecGenerator(dspy.Module):
    """Generate a raw JSON string spec using a single DSPy predictor."""

    def __init__(self, temperature: float = 0.0, verbose: bool = True):
        super().__init__()
        # Using a simple prompt → output signature keeps things explicit.
        self.generator = dspy.Predict("prompt -> spec_json")
        self.temperature = temperature
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[DSPy][Generate] {message}")

    def _compose_prompt(self, nl_request: str, available_udfs: Iterable[str]) -> str:
        # TODO: We can probably make this an optimizer task (like GEPA)
        formatted = sorted(available_udfs)
        if formatted:
            available = "\n  - " + "\n  - ".join(formatted)
        else:
            available = " (none)"
        header = PROMPT_HEADER.format(available_udfs=available)
        fewshot = _json_dumps(FEWSHOT_JSON)
        return (
            f"{header}\n\n"
            f"Example NL description:\n{FEWSHOT_USER}\n\n"
            f"Example JSON spec:\n{fewshot}\n\n"
            "Now respond to the new request. Return JSON ONLY."
            f"\n\nUser request:\n{nl_request}\n"
        )

    def forward(self, nl_request: str, available_udfs: Iterable[str]):
        prompt = self._compose_prompt(nl_request, available_udfs)
        self._log("Submitting prompt to language model...\n\n"+prompt)
        prediction = self.generator(prompt=prompt)
        self._log("Received raw response from language model")
        spec_json = prediction.spec_json
        if isinstance(spec_json, dict):
            # Some LMs may already return parsed dicts, normalize to string.
            spec_json = _json_dumps(spec_json)
            self._log("Converted structured response into JSON string")
        return spec_json


class SpecSemanticChecker:
    """Mirror the semantic checker used in the LangChain pipeline."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[DSPy][Semantic] {message}")

    def __call__(self, spec: QuerySpec) -> QuerySpec:
        self._log("Validating predicates against registry …")
        for keyframe in spec.keyframes:
            for atom in collect_atoms(keyframe.where):
                if atom.type not in GLOBAL_UDF_REGISTRY.get_all_udfs():
                    self._log(f"Auto-registering new predicate '{atom.type}'")
                    func = GLOBAL_UDF_REGISTRY.autogen_udf(atom.type)
                    GLOBAL_UDF_REGISTRY.register_udf(atom.type, func)
        self._log("Semantic validation complete")
        return spec


def collect_atoms(expr: PredicateExpr) -> list[PredicateAtom]:
    if expr.op == "ATOM" and expr.atom is not None:
        return [expr.atom]
    atoms: list[PredicateAtom] = []
    if expr.args:
        for sub in expr.args:
            atoms.extend(collect_atoms(sub))
    return atoms


class SpecParser:
    """Parse JSON into `QuerySpec`, raising helpful errors."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[DSPy][Parse] {message}")

    def __call__(self, spec_json: str) -> QuerySpec:
        self._log("Parsing JSON into QuerySpec …")
        try:
            raw = json.loads(spec_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Generated spec is not valid JSON: {exc}") from exc

        try:
            spec = QuerySpec.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Generated spec failed schema validation: {exc}") from exc

        self._log("Schema validation succeeded")
        return spec


class NLToQuerySpecPipeline(dspy.Module):
    """High-level module wiring generation, parsing, and semantic checking."""

    def __init__(self, *, temperature: float = 0.0, verbose: bool = True):
        super().__init__()
        self.verbose = verbose
        self.generator = SpecGenerator(temperature=temperature, verbose=verbose)
        self.parser = SpecParser(verbose=verbose)
        self.semantic_checker = SpecSemanticChecker(verbose=verbose)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[DSPy][Pipeline] {message}")

    def forward(
        self,
        nl_request: str,
        available_udfs: Optional[Iterable[str]] = None,
    ) -> QuerySpec:
        self._log("Starting NL → QuerySpec translation")
        if available_udfs is None:
            requested_udfs: list[object] = list(GLOBAL_UDF_REGISTRY.get_all_udfs().keys())
        else:
            requested_udfs = list(available_udfs)

        available_names = sorted({_udf_name(item) for item in requested_udfs})
        available_for_prompt = _format_available_udfs_for_prompt(requested_udfs)

        self._log(
            f"Using {len(available_names)} predicates: {', '.join(available_names)}"
        )
        raw_json = self.generator(
            nl_request=nl_request,
            available_udfs=available_for_prompt,
        )
        self._log("Generation complete, parsing …")
        spec = self.parser(raw_json)
        self._log("Running semantic checks …")
        validated = self.semantic_checker(spec)
        self._log("Pipeline finished")
        return validated


def build_pipeline(*, temperature: float = 0.0) -> NLToQuerySpecPipeline:
    """Helper that returns a ready-to-use pipeline."""

    return NLToQuerySpecPipeline(temperature=temperature)


def _mask_secret(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


def configure_lm(model: str, **kwargs) -> None:
    """Convenience wrapper around `dspy.configure` for LM setup."""

    display_kwargs = {k: (_mask_secret(v) if "key" in k.lower() else v) for k, v in kwargs.items()}
    print(f"[DSPy] Configuring LM → model={model}, kwargs={display_kwargs}")

    lm = dspy.LM(model=model, **kwargs, max_tokens=20_000)
    dspy.configure(lm=lm)


@dataclass
class PipelineResult:
    """Bundle returned from the CLI helper for easier inspection."""

    spec: QuerySpec
    spec_json: str


def run_pipeline(nl_request: str, pipeline: Optional[NLToQuerySpecPipeline] = None) -> PipelineResult:
    """Utility for scripts/tests to obtain both the spec object and JSON string."""

    pipeline = pipeline or build_pipeline()
    available = _format_available_udfs_for_prompt(
        GLOBAL_UDF_REGISTRY.get_all_udfs().keys()
    )
    print("[DSPy][Run] Executing pipeline with default configuration …")
    raw_json = pipeline.generator(nl_request=nl_request, available_udfs=available)
    spec = pipeline.parser(raw_json)
    spec = pipeline.semantic_checker(spec)
    print("[DSPy][Run] Done")
    return PipelineResult(spec=spec, spec_json=raw_json)


def write_spec_pickle(spec: QuerySpec, path: str) -> None:
    """Persist a QuerySpec object to disk so NL/main.py can load it."""

    with open(path, "wb") as handle:
        pickle.dump(spec, handle)
    print(f"[DSPy][Persist] Wrote pickle spec to {path}")


