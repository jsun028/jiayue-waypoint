# specs.py
from __future__ import annotations
from pydantic import BaseModel, field_validator, Field
from typing import List, Literal, Optional, Tuple, Dict, Union, Any

class PredicateAtom(BaseModel):
    # TODO: Define these types more explicitly
    type: Literal[
        "heading_diff_to", "velocity_above", "velocity_below",
        "speed_above", "speed_below", "dist_within", "dist_apart",
        "within_bbox", "action"
    ]
    # lhs object alias used within the spec, e.g. "car1"
    obj: str
    # optional rhs target (for pairwise predicates)
    other_obj: Optional[str] = None
    # numeric params (angles/thresholds/boxes)
    value: Optional[float] = None
    tol: Optional[float] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    label: Optional[str] = None  # e.g. action label

class PredicateExpr(BaseModel):
    # Boolean expression is expressed as a tree (AND/OR/NOT) over atoms.
    op: Literal["ATOM", "AND", "OR", "NOT"]
    atom: Optional[PredicateAtom] = None
    args: Optional[List["PredicateExpr"]] = None

PredicateExpr.update_forward_refs()

class KeyframeSpec(BaseModel):
    name: str
    where: PredicateExpr

class AlwaysSpec(BaseModel):
    kind: Literal["always"] = "always"
    anchor: Optional[str] = None         # KF name or None (self-anchored)
    target: str                          # KF name
    duration_sec: float = 3.0
    tol: float = 0.0

class EventuallySpec(BaseModel):
    kind: Literal["eventually"] = "eventually"
    anchor: str
    target: str
    duration_sec: float

class InterframeSpec(BaseModel):
    kind: Literal["interframe"] = "interframe"
    anchor: str
    target: str
    time_shift: float
    comparators: List[Dict[str, Any]]  # e.g. [{"type":"heading_diff","value":90.0,"tol":15.0}]

class TrajectorySpec(BaseModel):
    kind: Literal["trajectory"] = "trajectory"
    obj: str             # "car1"
    start: str           # KF name
    end: str             # KF name
    template: str        # e.g. "right_arc"
    angle_rad: float     # e.g. np.pi*0.75
    deviation_strength: float = 0.0

ConstraintSpec = Union[AlwaysSpec, EventuallySpec, InterframeSpec, TrajectorySpec]

class ObjectsSpec(BaseModel):
    # {"car":2} + an alias map like {"car1":{"class":"car","idx":0}, "car2":{"class":"car","idx":1}}
    # TODO: Figure out if this is the best way to represent this...
    counts: Dict[str, int]
    aliases: Dict[str, Dict[str, Any]]   # alias -> {"class":..., "idx":...}

class QuerySpec(BaseModel):
    objects: ObjectsSpec
    keyframes: List[KeyframeSpec]
    constraints: List[ConstraintSpec]

    @field_validator("keyframes")
    @classmethod
    def kf_names_unique(cls, v):
        names = [k.name for k in v]
        assert len(names) == len(set(names)), "Duplicate keyframe names"
        return v
