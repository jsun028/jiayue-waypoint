# specs.py
from __future__ import annotations
from pydantic import BaseModel, field_validator, Field, model_validator
from typing import List, Literal, Optional, Tuple, Dict, Union, Any
from .registry import UDFRegistry, GLOBAL_UDF_REGISTRY
import json


class PredicateAtom(BaseModel):
    # TODO: can we connect this to UDFRegistry dynamically?
    # see semantic_checker in experiments.py.
    # type: Literal[
    #     "velocity_above", "velocity_below", 
    #     "dist_within_two_obj", "dist_apart_two_obj", 
    #     "is_approaching", "is_separating", 
    #     "heading_diff_to"
    # ]
    type: str
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
    explanation: Optional[str] = None
    objects: ObjectsSpec
    keyframes: List[KeyframeSpec]
    constraints: List[ConstraintSpec]

    # True -> use combinations, False -> use assignments
    # False -> use permutations
    use_combinations: bool = False 

    @field_validator("keyframes")
    @classmethod
    def kf_names_unique(cls, v):
        names = [k.name for k in v]
        assert len(names) == len(set(names)), "Duplicate keyframe names"
        return v

    
def print_spec_details(spec: QuerySpec):

    print(f"Objects: {spec.objects.counts}")
    print(f"Keyframes: {len(spec.keyframes)}")
    print(f"Constraints: {len(spec.constraints)}")
    # Show keyframe details
    for i, kf in enumerate(spec.keyframes):
        print(f"Keyframe {i+1} ({kf.name}):")
        print(f"  Predicates: {kf.where.op if kf.where.op else kf.where.args}")
        if kf.where.op == "ATOM":
            print(f"    Atom details: Type: {kf.where.atom.type}, Object: {kf.where.atom.obj}, Other object: {kf.where.atom.other_obj}, Value: {kf.where.atom.value}")
        elif kf.where.args:
            for j, arg in enumerate(kf.where.args):
                if arg.op == "ATOM":
                    print(f"    {j+1}. Atom details: Type: {arg.atom.type}, Object: {arg.atom.obj}, Other object: {arg.atom.other_obj}, Value: {arg.atom.value}")
    
    # Show constraint details
    for i, c in enumerate(spec.constraints):
        print(f"Constraint {i+1} ({c.kind}):")
        if c.kind == "always":
            print(f"  Target: {c.target}, Duration: {c.duration_sec}s")
        elif c.kind == "interframe":
            print(f"  {c.anchor} -> {c.target}, Time shift: {c.time_shift}s, Comparators: {c.comparators}")
        elif c.kind == "trajectory":
            print(f"  Object: {c.obj}, {c.start} -> {c.end}, Template: {c.template}")
    
    print()

    # @model_validator(mode="after")
    # def kf_semantic_unique(self) -> "QuerySpec":
    #     seen = {}
    #     for kf in self.keyframes:
    #         # compare semantic equivalence without 'name'
    #         where_dict = kf.where.model_dump()
    #         key = json.dumps(where_dict, sort_keys=True)  # stable comparison
    #         if key in seen:
    #             raise ValueError(
    #                 f"Duplicate keyframe content detected: "
    #                 f"{kf.name} is semantically identical to {seen[key]}"
    #             )
    #         seen[key] = kf.name
    #     return self

    # no validation for explanation; human-readable only
