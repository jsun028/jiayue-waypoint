"""DSPy implementation of the NL → keyframe query specification pipeline."""

from __future__ import annotations

import inspect
import json
import os
import pickle
from dataclasses import dataclass
from typing import Iterable, Optional

import dspy

from keyframeql.registry import GLOBAL_UDF_REGISTRY
from keyframeql.specs import (
    QuerySpec,
    PredicateAtom,
    PredicateExpr
)
from keyframeql.spec_parser import SpecParser


PROMPT_HEADER = """TASK: Translate traffic scene descriptions into JSON specs for the keyframe query engine.
- Output ONLY valid JSON; no markdown code fences or commentary.
- Define keyframes as salient frames that capture important transitions/events.
- Name keyframes k1, k2, k3, ... in temporal order.
- Use object aliases (car1, car2, pedestrian1, ...) unless NL input specifies otherwise.
- Use degrees for angles unless NL explicitly requests radians.
- For "right turn" events, you may add a trajectory constraint with template="right_arc" (optional guidance only).
- Set use_combinations=true to assign unique sets of tracks per class (ignore alias permutations).
- Ego pose is represented by dedicated rows where class_name=="ego" or track_id==0; if you include an alias with class "ego", it refers to that track and is pre-bound by the engine (not enumerated).

PREDICATE SYSTEM: 

Two styles are available: MONOLITHIC (single function) or COMPOSITIONAL (computation + operator).

MONOLITHIC STYLE:
Use when an existing predicate function exactly matches your query.
Single function performs both computation and comparison.
Format: {"type": "car_turning", "obj": "car1", "value": 2.0, "mode": "left"}

STYLE 2: COMPOSITIONAL (Computation + Operator)
Separates raw value computation from threshold comparison for flexibility.
Use when: No existing monolithic predicate matches, or you need custom thresholds/operators.

Structure: Operator(Computation(args), threshold)
- Computation functions (return raw pd.Series values): distance, velocity, heading_diff, rotational_velocity, acceleration, closing_speed, visibility_score, relative_position_local
- Operator function: LessThan, GreaterThan, InRange, SoftClose, Equal

Example patterns:
- LessThan(distance(car1, car2), 50.0) → {"type": "LessThan", "computation": {"type": "distance", "obj": "car1", "other_obj": "car2"}, "value": 50.0}
- GreaterThan(velocity(car1), 5.0) → {"type": "GreaterThan", "computation": {"type": "velocity", "obj": "car1"}, "value": 5.0}
- SoftClose(heading_diff(car1, car2), 90.0, 30.0) → {"type": "SoftClose", "computation": {"type": "heading_diff", "obj": "car1", "other_obj": "car2"}, "value": 90.0, "tol": 30.0}
- InRange(velocity(car1), 2.0, 8.0) → {"type": "InRange", "computation": {"type": "velocity", "obj": "car1"}, "value": 2.0, "tol": 8.0}

JSON format:
{"type": "GreaterThan", "computation": {"type": "velocity", "obj": "car1"}, "value": 5.0}

DISCRETE SLIDERS

Use DiscreteSlider objects for numeric parameters (value, tol) to enable runtime tuning.
Three settings: low (most selective), medium (balanced), high (most permissive).
The "medium" value should be your best estimate; adjust low/high to create a useful tuning range
Format: {"low": X, "medium": Y, "high": Z}

CRITICAL RULE: Slider keys refer to SELECTIVITY (how many results), not numeric values.
- low = fewest results (most selective)
- medium = balanced
- high = most results (most permissive)

Direction rules by operator:
- LessThan(X): smaller threshold = more selective 
- GreaterThan(X): larger threshold = more selective 
- InRange(min, max): narrow range = more selective → widen range from low to high
- SoftClose(target, cutoff): larger cutoff = more permissive. target (domain-specific) doesn't affect selectivity much

Ask: "Does a LARGER number make the predicate EASIER to satisfy?"
- Easier → ascending values (low < medium < high)
- Harder → descending values (low > medium > high)

CONSTRUCTION GUIDE
- Each predicate shows "Spec: PredicateAtom(...)" indicating how to build it
- Parameters in angle brackets (e.g., <velocity>) should be replaced with actual values. (e.g. in dist_within_two_obj(), value=<distance>, value is a key and distance is a value)
- "obj" field: use object alias from your spec (e.g., "car1", "pedestrian1")
- "other_obj" field: for pairwise predicates, use second object alias
- "frame_window" is handled automatically by the query engine (set to None in PredicateAtom)
- Map function parameters to PredicateAtom fields as shown in each spec line

Available predicates (each shows function signature and PredicateAtom construction):
{available_udfs}

CONSTRAINTS

Types:
- always: Predicate holds continuously for duration
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

Other guidance:
- where clause in JSON format should use "args" instead of "arg" no matter how long the list is
- Always include a concise "explanation" string that summarizes the reasoning behind the design of objects, keyframes, and constraints.
- Keep the number of objects minimal for the story; use the ego alias only when the NL explicitly refers to the ego vehicle (e.g., "ego nearly hits pedestrian").
- Remember to use DiscreteSlider objects for all numeric thresholds and tolerances to enable post-generation tuning.

CRITICAL - Avoiding Over-Constrained Queries:
- Each predicate multiplies selectivity - more predicates = exponentially fewer matches
- EARLY keyframes (k1) should be SIMPLER and BROADER to ensure matches exist before refining
- Limit k1 to 3 predicates maximum; use the most essential conditions only
- For multi-keyframe specs (3+), keep each keyframe to 2-3 predicates
- Use BROAD slider ranges in early keyframes: "high" value should be quite permissive
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
                            "type": "SoftClose",
                            "computation": {
                                "type": "heading_diff",
                                "obj": "car1",
                                "other_obj": "car2"
                            },
                            "value": {"low": 170.0, "medium": 180.0, "high": 190.0},
                            "tol": {"low": 10.0, "medium": 15.0, "high": 20.0},
                        },
                    },
                    {
                        "op": "ATOM",
                        "atom": {
                            "type": "GreaterThan",
                            "computation": {
                                "type": "velocity",
                                "obj": "car1"
                            },
                            "value": {"low": 2.5, "medium": 1.5, "high": 0.5}
                        },
                    },
                    {
                        "op": "ATOM",
                        "atom": {
                            "type": "GreaterThan",
                            "computation": {
                                "type": "velocity",
                                "obj": "car2"
                            },
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
                            "type": "SoftClose",
                            "computation": {
                                "type": "heading_diff",
                                "obj": "car1",
                                "other_obj": "car2"
                            },
                            "value": {"low": 85.0, "medium": 90.0, "high": 95.0},
                            "tol": {"low": 10.0, "medium": 15.0, "high": 20.0},
                        },
                    },
                    {
                        "op": "ATOM",
                        "atom": {
                            "type": "GreaterThan",
                            "computation": {
                                "type": "velocity",
                                "obj": "car1"
                            },
                            "value": {"low": 2.5, "medium": 1.5, "high": 0.5}
                        },
                    },
                    {
                        "op": "ATOM",
                        "atom": {
                            "type": "GreaterThan",
                            "computation": {
                                "type": "velocity",
                                "obj": "car2"
                            },
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
            "comparators": [],
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
        "car1 and car2 begin opposed while moving (180° heading difference), then car1 slows to set up a right turn, "
        "and over ~3s completes a right-arc trajectory to align perpendicular (90°) to car2. "
        "Uses compositional predicates: SoftClose for fuzzy heading match, GreaterThan for velocity thresholds."
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
                    {
                        "op": "ATOM", 
                        "atom": {
                            "type": "SoftClose",
                            "computation": {
                                "type": "heading_diff",
                                "obj": "car1",
                                "other_obj": "pedestrian1"
                            },
                            "value": 90.0,
                            "tol": 10.0
                        }
                    },
                    {
                        "op": "ATOM", 
                        "atom": {
                            "type": "car_can_see_agent", 
                            "obj": "car1", 
                            "other_obj": "pedestrian1", 
                            "value": 45.0, 
                            "tol": 10.0
                        }
                    },
                    {
                        "op": "ATOM", 
                        "atom": {
                            "type": "LessThan",
                            "computation": {
                                "type": "distance",
                                "obj": "car1",
                                "other_obj": "pedestrian1"
                            },
                            "value": 30.0
                        }
                    },
                    {
                        "op": "ATOM", 
                        "atom": {
                            "type": "GreaterThan",
                            "computation": {
                                "type": "velocity",
                                "obj": "car1"
                            },
                            "value": 1.0
                        }
                    },
                ],
            },
        },
        {
            "name": "k2",
            # BAD: Extreme distance threshold (1-3m is near-collision, very rare)
            "where": {
                "op": "ATOM", 
                "atom": {
                    "type": "LessThan",
                    "computation": {
                        "type": "distance",
                        "obj": "car1",
                        "other_obj": "pedestrian1"
                    },
                    "value": {"low": 1.0, "medium": 2.0, "high": 3.0}
                }
            },
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


@dataclass
class SpecSession:
    """Tracks conversation history for iterative refinement."""
    nl_request: str
    history: list[dict]  # [{"spec": QuerySpec, "spec_json": str, "feedback": str}]
    current_spec: Optional[QuerySpec] = None
    current_spec_json: Optional[str] = None
    
    def add_iteration(self, spec: QuerySpec, spec_json: str, feedback: Optional[str] = None):
        """Record a refinement iteration."""
        self.history.append({
            "spec": spec,
            "spec_json": spec_json,
            "feedback": feedback
        })
        self.current_spec = spec
        self.current_spec_json = spec_json

def start_session(nl_request: str, pipeline: Optional[NLToQuerySpecPipeline] = None, 
                  stats_text: Optional[str] = None) -> SpecSession:
    """Start a new spec generation session."""
    pipeline = pipeline or build_pipeline()
    
    session = SpecSession(nl_request=nl_request, history=[])
    spec = pipeline(nl_request, stats_text=stats_text, session=session)
    
    return session


def refine_spec(session: SpecSession, feedback: str, 
                pipeline: Optional[NLToQuerySpecPipeline] = None,
                stats_text: Optional[str] = None) -> QuerySpec:
    """Refine the current spec based on user feedback."""
    pipeline = pipeline or build_pipeline()
    
    spec = pipeline(
        session.nl_request, 
        stats_text=stats_text,
        session=session,
        feedback=feedback
    )
    
    return spec

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
        previous_spec: Optional[str] = None,  # NEW
        feedback: Optional[str] = None,       # NEW
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
        
        # Add refinement context if this is an iteration
        refinement_context = ""
        if previous_spec and feedback:
            refinement_context = (
                f"\n\n=== REFINEMENT MODE ===\n"
                f"You previously generated this spec:\n{previous_spec}\n\n"
                f"User feedback:\n{feedback}\n\n"
                f"Modify the spec to address the feedback while preserving aspects that weren't criticized. "
                f"Return the complete updated JSON spec.\n"
                f"=== END REFINEMENT CONTEXT ===\n"
            )
        
        base_instruction = (
            "Now respond to the new request. Return JSON ONLY. Follow the GOOD pattern, avoid the BAD pattern."
            if not refinement_context
            else "Return the complete refined JSON spec addressing the feedback."
        )
        
        return (
            f"{header}\n\n"
            f"GOOD EXAMPLE - NL description:\n{FEWSHOT_USER}\n\n"
            f"GOOD EXAMPLE - JSON spec (FOLLOW THIS PATTERN):\n{fewshot_good}\n\n"
            f"BAD EXAMPLE - Over-constrained spec (AVOID THIS PATTERN):\n{fewshot_bad}"
            f"{stats_block}"
            f"{refinement_context}\n\n"
            f"{base_instruction}"
            f"\n\nOriginal user request:\n{nl_request}\n"
        )

    def forward(
        self,
        nl_request: str,
        available_udfs: Iterable[str],
        *,
        stats_text: Optional[str] = None,
        previous_spec: Optional[str] = None,  
        feedback: Optional[str] = None,     
    ):
        prompt = self._compose_prompt(
            nl_request, 
            available_udfs, 
            stats_text=stats_text,
            previous_spec=previous_spec,
            feedback=feedback
        )
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
        session: Optional[SpecSession] = None,
        feedback: Optional[str] = None
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
        if session and feedback:
            self._log(f"Refinement mode: iteration {len(session.history) + 1}")
        else:
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

        # NEW: Pass previous spec if refining
        previous_spec = session.current_spec_json if session else None

        raw_json = self.generator(
            nl_request=nl_request,
            available_udfs=available_for_prompt,
            stats_text=stats_text,
            previous_spec=previous_spec,
            feedback=feedback,
        )
        self._log("Generation complete, parsing …")
        spec = self.parser(raw_json)
        self._log("Running semantic checks …")
        validated = self.semantic_checker(spec)
        
        if session:
            session.add_iteration(validated, raw_json, feedback)

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


