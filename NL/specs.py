# specs.py
from __future__ import annotations
from pydantic import BaseModel, field_validator, Field
from typing import List, Literal, Optional, Tuple, Dict, Union, Any
from registry import UDFRegistry


class PredicateAtom(BaseModel):
    # TODO: can we connect this to UDFRegistry dynamically?
    type: Literal["velocity_above", "velocity_below", "dist_within_two_obj",         
            "speed_above", "speed_below",  "dist_apart",         
            "within_bbox", "action" ]
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

PredicateExpr.model_rebuild()

class KeyframeSpec(BaseModel):
    name: str
    where: PredicateExpr

class AlwaysSpec(BaseModel):
    kind: Literal["always"] = "always"
    anchor: Optional[str] = None         # KF name or None (self-anchored)
    target: str                          # KF name
    duration_sec: float = 3.0
    tol: float = 0.0

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

ConstraintSpec = Union[AlwaysSpec, InterframeSpec, TrajectorySpec]

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
