"""DSPy implementation of the NL → keyframe query specification pipeline."""

from __future__ import annotations

import inspect
import json
import os
import pickle
from dataclasses import dataclass
from typing import Iterable, Optional, Any, Union

import dspy

from NL.registry import GLOBAL_UDF_REGISTRY
from NL.specs import (
    QuerySpec,
    PredicateAtom,
    PredicateExpr,
    KeyframeSpec,
    ObjectsSpec,
    AlwaysSpec,
    InterframeSpec,
    TrajectorySpec,
    ComputationSpec,
)


PROMPT_HEADER = """You translate traffic scene descriptions into JSON specs for the keyframe query engine.
- Define keyframes as salient frames that capture important transitions/events.
- Name keyframes k1, k2, k3, ... in temporal order.
- Output ONLY valid JSON; no markdown code fences or commentary.
- Use object aliases (car1, car2, pedestrian1, ...) unless NL input specifies otherwise.
- Use degrees for angles unless NL explicitly requests radians.
- For "right turn" events, you may add a trajectory constraint with template="right_arc" (optional guidance only).
- Set use_combinations=true to assign unique sets of tracks per class (ignore alias permutations).
- Ego pose is represented by dedicated rows where class_name=="ego" or track_id==0; if you include an alias with class "ego", it refers to that track and is pre-bound by the engine (not enumerated).

COMPOSITIONAL PREDICATE SYSTEM:
The system supports two predicate styles:

1. MONOLITHIC (legacy): Single function does computation + comparison
   Example: {"type": "velocity_above", "obj": "car1", "value": 10.0}
   
2. COMPOSITIONAL (preferred): Separate computation and operator
   Example: {"type": "GreaterThan", "computation": {"type": "velocity", "obj": "car1"}, "value": 10.0}
   
Compositional style is MORE FLEXIBLE and composable:
- Computation functions return raw values: distance, velocity, heading_diff, rotational_velocity, acceleration
- Operator functions score the values: LessThan, GreaterThan, InRange, SoftClose, Equal

Example patterns:
- LessThan(distance(car1, car2), 50.0) → {"type": "LessThan", "computation": {"type": "distance", "obj": "car1", "other_obj": "car2"}, "value": 50.0}
- GreaterThan(velocity(car1), 5.0) → {"type": "GreaterThan", "computation": {"type": "velocity", "obj": "car1"}, "value": 5.0}
- SoftClose(heading_diff(car1, car2), 90.0, 30.0) → {"type": "SoftClose", "computation": {"type": "heading_diff", "obj": "car1", "other_obj": "car2"}, "value": 90.0, "tol": 30.0}
- InRange(velocity(car1), 2.0, 8.0) → {"type": "InRange", "computation": {"type": "velocity", "obj": "car1"}, "value": 2.0, "tol": 8.0}

USE COMPOSITIONAL STYLE for new predicates unless you need legacy UDF compatibility.

Discrete Sliders for Tunable Values:
- For numeric parameters like "value" and "tol" (tolerance), you SHOULD use DiscreteSlider objects instead of plain floats.
- A DiscreteSlider has three settings that users select at runtime to control query selectivity
- User can run with --slider-setting=low (most selective), --slider-setting=medium (balanced), or --slider-setting=high (most permissive)
- The "medium" value should be your best estimate; adjust low/high to create a useful tuning range

CRITICAL - Slider Direction Based on Predicate Semantics:
The slider keys "low"/"medium"/"high" refer to SELECTIVITY, not the numeric values!
- "low" = most SELECTIVE (fewest results)
- "medium" = balanced
- "high" = most PERMISSIVE (most results)

For "above/below" threshold predicates:
- velocity_above(X): smaller X is easier to exceed (MORE results)
  → {"low": 2.0, "medium": 1.0, "high": 0.5}  ← "high" selectivity uses small threshold
- velocity_below(X): larger X is easier to stay under (MORE results)
  → {"low": 0.05, "medium": 0.1, "high": 0.2}

For distance predicates:
- dist_within_two_obj(X): larger X allows more distant pairs (MORE results)
  → {"low": 50.0, "medium": 100.0, "high": 200.0}

For angle predicates:
- heading_diff(value=A, tol=T): larger T accepts more variation (MORE results)
  → value: {"low": 85, "medium": 90, "high": 95}  ← target angle doesn't affect selectivity much
  → tol: {"low": 10, "medium": 20, "high": 30}    ← larger tolerance = more permissive

Rule of thumb:
- Ask: "Does a LARGER number make the predicate EASIER or HARDER to satisfy?"
- Easier → use ascending values (low < medium < high)
- Harder → use descending values (low > medium > high)

Available predicates (each shows function signature and PredicateAtom construction):
{available_udfs}

Constraint semantics (what the engine enforces):
- always:
  * Self-anchored: {"kind":"always", "anchor": null, "target": "kX", "duration_sec": D}
    - Interpreted as: when evaluating keyframe kX at frame t, kX must hold continuously on [t, t+D].
  * Cross-anchored: {"kind":"always", "anchor": "kA", "target": "kB", "duration_sec": D}
    - Interpreted as: after kA occurs at frame tA, kB must be satisfied for all frames in [tA, tA+D].
- interframe:
  * {"kind":"interframe", "anchor": "kA", "target": "kB", "time_shift": S, "comparators": []}
    - Interpreted as: the time difference between kA and kB is approximately S seconds.
    - Timing tolerance: ±0.1 seconds.
    - Comparators are reserved for future use; omit or leave empty.
- Keyframe ordering: the engine only accepts strictly increasing keyframes in results (k1 < k2 < ...). Gaps must be within reasonable bounds.

PredicateAtom construction guide:
- Each predicate shows "Spec: PredicateAtom(...)" indicating how to build it
- Parameters in angle brackets (e.g., <velocity>) should be replaced with actual values. (e.g. in dist_within_two_obj(), value=<distance>, value is a key and distance is a value)
- "obj" field: use object alias from your spec (e.g., "car1", "pedestrian1")
- "other_obj" field: for pairwise predicates, use second object alias
- "frame_window" is handled automatically by the query engine (set to None in PredicateAtom)
- Map function parameters to PredicateAtom fields as shown in each spec line

- You may propose a new predicate only if absolutely required.
- Always include a concise "explanation" string that summarizes the reasoning behind the design of objects, keyframes, and constraints.
- Do not use interframe constraints yet, they do not work.

Other guidance:
- Keep the number of objects minimal for the story; use the ego alias only when the NL explicitly refers to the ego vehicle (e.g., "ego nearly hits pedestrian").
- Prefer agent-ego predicates (e.g., heading_diff_agent_to_ego) when relating an agent to the ego; use agent-agent predicates for agent pairs.
- Remember to use DiscreteSlider objects for all numeric thresholds and tolerances to enable post-generation tuning.

CRITICAL - Avoiding Over-Constrained Queries:
- Each predicate multiplies selectivity - more predicates = exponentially fewer matches
- EARLY keyframes (k1) should be SIMPLER and BROADER to ensure matches exist before refining
- Limit k1 to 2-4 predicates maximum; use the most essential conditions only
- For multi-keyframe specs (3+), keep each keyframe to 2-3 predicates
- Use BROAD slider ranges in early keyframes: "high" value should be quite permissive
- Distance thresholds: use 50-300m for proximity (not <10m unless describing near-collision)
- Angle tolerances: use at least ±15-30° tolerance in "medium" setting
- Avoid combining multiple geometric predicates (heading + visibility + distance) in k1
- Build narrative progression: k1 = setup (broad), k2 = development (medium), k3 = climax (tighter)
- If a spec has N keyframes, the probability of finding a match is roughly P₁ × P₂ × ... × Pₙ where each Pᵢ < 1
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
    "use_combinations": True,
    "keyframes": [
        {
            "name": "k1",
            "where": {
                "op": "AND",
                "args": [
                    {
                        "op": "ATOM",
                        "atom": {
                            "type": "heading_diff_agent_to_agent",
                            "obj": "car1",
                            "other_obj": "car2",
                            "value": {"low": 170.0, "medium": 180.0, "high": 190.0},
                            "tol": {"low": 10.0, "medium": 15.0, "high": 20.0},
                        },
                    },
                    {
                        "op": "ATOM",
                        "atom": {
                            "type": "velocity_above", 
                            "obj": "car1", 
                            "value": {"low": 2.5, "medium": 1.5, "high": 0.5}
                        },
                    },
                    {
                        "op": "ATOM",
                        "atom": {
                            "type": "velocity_above", 
                            "obj": "car2", 
                            "value": {"low": 2.5, "medium": 1.5, "high": 0.5}
                        },
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
                            "type": "heading_diff_agent_to_agent",
                            "obj": "car1",
                            "other_obj": "car2",
                            "value": {"low": 85.0, "medium": 90.0, "high": 95.0},
                            "tol": {"low": 10.0, "medium": 15.0, "high": 20.0},
                        },
                    },
                    {
                        "op": "ATOM",
                        "atom": {
                            "type": "velocity_above", 
                            "obj": "car1", 
                            "value": {"low": 2.5, "medium": 1.5, "high": 0.5}
                        },
                    },
                    {
                        "op": "ATOM",
                        "atom": {
                            "type": "velocity_above", 
                            "obj": "car2", 
                            "value": {"low": 2.5, "medium": 1.5, "high": 0.5}
                        },
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
            "comparators": [{"type": "heading_diff_agent_to_agent", "value": 90.0, "tol": 15.0}],
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

# ANTI-PATTERN: Over-constrained spec (AVOID THIS)
FEWSHOT_BAD_JSON = {
    "objects": {"counts": {"car": 1, "pedestrian": 1}},
    "use_combinations": True,
    "keyframes": [
        {
            "name": "k1",
            "where": {
                "op": "AND",
                "args": [
                    # BAD: Too many geometric constraints in k1
                    {"op": "ATOM", "atom": {"type": "heading_diff_agent_to_agent", "obj": "car1", "other_obj": "pedestrian1", "value": 90.0, "tol": 10.0}},
                    {"op": "ATOM", "atom": {"type": "car_can_see_agent", "obj": "car1", "other_obj": "pedestrian1", "value": 45.0, "tol": 10.0}},
                    {"op": "ATOM", "atom": {"type": "dist_within_two_obj", "obj": "car1", "other_obj": "pedestrian1", "value": 30.0}},
                    {"op": "ATOM", "atom": {"type": "velocity_above", "obj": "car1", "value": 1.0}},
                ],
            },
        },
        {
            "name": "k2",
            # BAD: Extreme distance threshold (1-3m is near-collision, very rare)
            "where": {"op": "ATOM", "atom": {"type": "dist_within_two_obj", "obj": "car1", "other_obj": "pedestrian1", "value": {"low": 1.0, "medium": 2.0, "high": 3.0}}},
        },
    ],
    "constraints": [],
    "explanation": "ANTI-PATTERN: k1 has 4 geometric predicates (too restrictive), k2 has extreme proximity (1-3m). This will likely return zero results.",
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
    
    # Add PredicateAtom mapping if available (from @udf decorator)
    # Always show spec line for consistency, even if no param mapping
    sig_params = list(signature.parameters.keys()) if signature != "(…)" else []
    
    if sig_params or hasattr(func, '_udf_param_mapping'):
        # Build the spec line showing how params map to PredicateAtom fields
        spec_parts = []
        
        # Add obj parameter (first object ID)
        spec_parts.append("obj=<object_id/oid1>")
        
        # Check if it's a pairwise predicate (has oid2 in signature)
        if 'oid2' in sig_params or any('oid2' in p.lower() for p in sig_params):
            spec_parts.append("other_obj=<oid2>")
        
        # Add parameter mappings from decorator
        if hasattr(func, '_udf_param_mapping'):
            param_mapping = func._udf_param_mapping
            for param_name, atom_attr in param_mapping.items():
                spec_parts.append(f"{atom_attr}=<{param_name}>")
        
        # Always include frame_window note
        # spec_parts.append("frame_window=<auto>")
        
        spec_line = f"Spec: PredicateAtom({', '.join(spec_parts)})"
        summary += f"\n    {spec_line}"
    
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


# Note: format_stats_for_prompt is imported from NL_dspy.stats_prompt


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

    def _compose_prompt(
        self,
        nl_request: str,
        available_udfs: Iterable[str],
        *,
        stats_text: Optional[str] = None,
    ) -> str:
        # TODO: We can probably make this an optimizer task (like GEPA)
        formatted = sorted(available_udfs)
        if formatted:
            available = "\n  - " + "\n  - ".join(formatted)
        else:
            available = " (none)"
        header = PROMPT_HEADER.replace("{available_udfs}", available)
        fewshot_good = _json_dumps(FEWSHOT_JSON)
        fewshot_bad = _json_dumps(FEWSHOT_BAD_JSON)
        stats_block = (
            f"\n\nDataset statistics (for grounding):\n{stats_text}\n"
            if stats_text
            else ""
        )
        return (
            f"{header}\n\n"
            f"GOOD EXAMPLE - NL description:\n{FEWSHOT_USER}\n\n"
            f"GOOD EXAMPLE - JSON spec (FOLLOW THIS PATTERN):\n{fewshot_good}\n\n"
            f"BAD EXAMPLE - Over-constrained spec (AVOID THIS PATTERN):\n{fewshot_bad}"
            f"{stats_block}\n\n"
            "Now respond to the new request. Return JSON ONLY. Follow the GOOD pattern, avoid the BAD pattern."
            f"\n\nUser request:\n{nl_request}\n"
        )

    def forward(
        self,
        nl_request: str,
        available_udfs: Iterable[str],
        *,
        stats_text: Optional[str] = None,
    ):
        prompt = self._compose_prompt(nl_request, available_udfs, stats_text=stats_text)
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
                # Check if compositional style
                if atom.computation is not None:
                    # Validate computation function
                    if atom.computation.type not in GLOBAL_UDF_REGISTRY.get_all_udfs():
                        self._log(f"Auto-registering new computation '{atom.computation.type}'")
                        func = GLOBAL_UDF_REGISTRY.autogen_udf(atom.computation.type)
                        GLOBAL_UDF_REGISTRY.register_udf(atom.computation.type, func)
                    
                    # Validate operator function
                    if atom.type not in GLOBAL_UDF_REGISTRY.get_all_udfs():
                        self._log(f"Auto-registering new operator '{atom.type}'")
                        func = GLOBAL_UDF_REGISTRY.autogen_udf(atom.type)
                        GLOBAL_UDF_REGISTRY.register_udf(atom.type, func)
                else:
                    # Monolithic style
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

    def _check_unknown_fields(
        self, data: Any, model_class: type, path: str = "root"
    ) -> None:
        """Recursively check for unknown fields in nested model structures."""
        if not isinstance(data, dict):
            return

        if not hasattr(model_class, "model_fields"):
            return

        allowed_fields = set(model_class.model_fields.keys())
        unknown_fields = set(data.keys()) - allowed_fields

        if unknown_fields:
            raise ValueError(
                f"Unknown fields in {path}: {sorted(unknown_fields)}. "
                f"Allowed fields are: {sorted(allowed_fields)}"
            )

        # Recursively check nested structures based on model fields
        for field_name, field_info in model_class.model_fields.items():
            if field_name not in data:
                continue

            field_value = data[field_name]
            field_type = field_info.annotation

            # Handle Optional types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                # Get the non-None type from Optional[Type] = Union[Type, None]
                non_none_types = [
                    t for t in field_type.__args__ if t is not type(None)
                ]
                if non_none_types:
                    field_type = non_none_types[0]
                else:
                    continue

            # Handle List types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                if isinstance(field_value, list):
                    item_type = (
                        field_type.__args__[0]
                        if hasattr(field_type, "__args__") and field_type.__args__
                        else None
                    )
                    # Handle specific list types
                    if field_name == "keyframes":
                        for i, kf in enumerate(field_value):
                            if isinstance(kf, dict):
                                self._check_unknown_fields(
                                    kf, KeyframeSpec, f"{path}.{field_name}[{i}]"
                                )
                    elif field_name == "constraints":
                        for i, constraint in enumerate(field_value):
                            if isinstance(constraint, dict):
                                kind = constraint.get("kind")
                                if kind == "always":
                                    self._check_unknown_fields(
                                        constraint,
                                        AlwaysSpec,
                                        f"{path}.{field_name}[{i}]",
                                    )
                                elif kind == "interframe":
                                    self._check_unknown_fields(
                                        constraint,
                                        InterframeSpec,
                                        f"{path}.{field_name}[{i}]",
                                    )
                                elif kind == "trajectory":
                                    self._check_unknown_fields(
                                        constraint,
                                        TrajectorySpec,
                                        f"{path}.{field_name}[{i}]",
                                    )
                    elif field_name == "args":
                        # Recursive PredicateExpr args
                        for i, arg in enumerate(field_value):
                            if isinstance(arg, dict):
                                self._check_unknown_fields(
                                    arg, PredicateExpr, f"{path}.{field_name}[{i}]"
                                )
                    elif item_type:
                        # Generic list handling
                        for i, item in enumerate(field_value):
                            if isinstance(item, dict):
                                self._check_unknown_fields(
                                    item, item_type, f"{path}.{field_name}[{i}]"
                                )
                continue

            # Handle BaseModel types
            if isinstance(field_value, dict):
                try:
                    # Try to identify the model class based on field name and structure
                    if field_name == "where" and field_value.get("op") is not None:
                        self._check_unknown_fields(
                            field_value, PredicateExpr, f"{path}.{field_name}"
                        )
                    elif field_name == "atom" and field_value.get("type") is not None:
                        self._check_unknown_fields(
                            field_value, PredicateAtom, f"{path}.{field_name}"
                        )
                    elif field_name == "computation" and field_value.get("type") is not None:
                        self._check_unknown_fields(
                            field_value, ComputationSpec, f"{path}.{field_name}"
                        )
                    elif field_name == "objects":
                        self._check_unknown_fields(
                            field_value, ObjectsSpec, f"{path}.{field_name}"
                        )
                except (AttributeError, TypeError):
                    # If we can't determine the type, skip deeper validation
                    pass

    def __call__(self, spec_json: str) -> QuerySpec:
        self._log("Parsing JSON into QuerySpec …")
        try:
            raw = json.loads(spec_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Generated spec is not valid JSON: {exc}") from exc

        try:
            # Recursively check for unknown fields at all levels
            if isinstance(raw, dict):
                self._check_unknown_fields(raw, QuerySpec, "QuerySpec")

            spec = QuerySpec.model_validate(raw)
        except ValueError as exc:
            # Re-raise ValueError as-is (our custom errors)
            raise exc
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
        *,
        stats_text: Optional[str] = None,
    ) -> QuerySpec:
        """Translate NL into a checked QuerySpec.

        Constraint interpretation in downstream compiler:
        - Self-anchored always: target keyframe must remain true over a contiguous window of
          duration_sec starting at the candidate frame for that keyframe.
        - Cross-anchored always: when anchor keyframe occurs at frame tA, target must hold for all
          frames in [tA, tA + duration_sec].
        - Interframe: target should occur approximately time_shift seconds after anchor, within
          ±0.1s tolerance (comparators are currently ignored).
        - Keyframes are enforced to be strictly increasing in time; extreme gaps are rejected.
        - If an alias with class "ego" is present, it is pre-bound to the ego track and excluded from
          object assignment enumeration.
        """
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
            stats_text=stats_text,
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

    # Default max_tokens, but allow override via kwargs
    # Some models (e.g., gpt-4o-mini) support max 16384 tokens
    max_tokens = kwargs.pop("max_tokens", 16_384)
    
    # Disable caching by default (allow override via kwargs)
    # Try multiple approaches to ensure caching is disabled
    caching = kwargs.pop("caching", False)
    # Set environment variable to disable LiteLLM caching
    if not caching:
        os.environ["LITELLM_DISABLE_CACHE"] = "true"
    
    # Pass caching parameter to dspy.LM
    lm = dspy.LM(model=model, max_tokens=max_tokens, caching=caching, **kwargs)
    dspy.configure(lm=lm)


@dataclass
class PipelineResult:
    """Bundle returned from the CLI helper for easier inspection."""

    spec: QuerySpec
    spec_json: str


def run_pipeline(
    nl_request: str,
    pipeline: Optional[NLToQuerySpecPipeline] = None,
    *,
    stats_text: Optional[str] = None,
) -> PipelineResult:
    """Utility for scripts/tests to obtain both the spec object and JSON string."""

    pipeline = pipeline or build_pipeline()
    available = _format_available_udfs_for_prompt(
        GLOBAL_UDF_REGISTRY.get_all_udfs().keys()
    )
    print("[DSPy][Run] Executing pipeline with default configuration …")
    raw_json = pipeline.generator(
        nl_request=nl_request,
        available_udfs=available,
        stats_text=stats_text,
    )
    print("\n=== Raw JSON ===")
    print(_json_dumps(json.loads(raw_json)))

    spec = pipeline.parser(raw_json)
    spec = pipeline.semantic_checker(spec)
    print("[DSPy][Run] Done")
    return PipelineResult(spec=spec, spec_json=raw_json)


def write_spec_pickle(spec: QuerySpec, path: str) -> None:
    """Persist a QuerySpec object to disk so NL/main.py can load it."""

    with open(path, "wb") as handle:
        pickle.dump(spec, handle)
    print(f"[DSPy][Persist] Wrote pickle spec to {path}")


