from pydantic import BaseModel
from typing import List, Optional, Literal, Any

class ObjectSchema(BaseModel):
    name: str
    cls: str
    idx: int

class PredicateSchema(BaseModel):
    op: str
    args: List[Any]
    kind: Literal["single", "inter"]

class KeyframeSchema(BaseModel):
    name: str
    predicates: List[PredicateSchema]

class TemporalAlwaysSchema(BaseModel):
    target: str
    dur_sec: float
    tol: Optional[float] = 0.01
    anchor: Optional[str] = None

class TemporalInterframeSchema(BaseModel):
    anchor: str
    target: str
    time_shift: float
    comparators: List[PredicateSchema]

class IRSchema(BaseModel):
    select: str
    dataset: str
    objects: List[ObjectSchema]
    keyframes: List[KeyframeSchema]
    temporal_always: List[TemporalAlwaysSchema]
    temporal_interframe: List[TemporalInterframeSchema]
    returns: List[str]
    query_name: str
