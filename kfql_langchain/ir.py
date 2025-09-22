from dataclasses import dataclass, field
from typing import List, Optional, Literal, Any

@dataclass
class IRObject:
    name: str
    cls: str
    idx: int

@dataclass
class IRPredicate:
    op: str
    args: List[Any]
    kind: Literal["single","inter"] = "single"

@dataclass
class IRKeyframe:
    name: str
    predicates: List[IRPredicate] = field(default_factory=list)

@dataclass
class IRTemporalAlways:
    target: str
    dur_sec: float
    tol: float = 0.01
    anchor: Optional[str] = None

@dataclass
class IRTemporalInterframe:
    anchor: str
    target: str
    time_shift: float
    comparators: List[IRPredicate]

@dataclass
class IRRoot:
    select: str
    dataset: str
    objects: List[IRObject]
    keyframes: List[IRKeyframe]
    temporal_always: List[IRTemporalAlways]
    temporal_interframe: List[IRTemporalInterframe]
    returns: List[str]
    query_name: str
