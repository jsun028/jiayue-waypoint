from ir import *
from typing import Any, List
from ir import IRRoot, IRPredicate
from schema import IRSchema  # ✅ Pydantic 기반 스키마

class SchemaError(Exception):
    pass

class TypeErrorKFQL(Exception):
    pass

class ConstraintError(Exception):
    pass

# operator definition (domain rules) → IRSchema and semantic validation
OP_SIGNATURES = {
    "velocity_above": {"kind": "single", "arity": 2},
    "velocity_below": {"kind": "single", "arity": 2},
    "heading_diff_to": {"kind": "single", "arity": 3},
    "distance_within": {"kind": "single", "arity": 2},
    "distance_apart": {"kind": "single", "arity": 2},
    "heading_diff": {"kind": "inter", "arity": 2},
}

"""
Pydantic schema: syntactic validation
checker.py     : semantic validation
"""

def _check_predicate(p: IRPredicate) -> None:
    """Check predicate against allowed signatures"""
    if p.op not in OP_SIGNATURES:
        raise SchemaError(f"Unknown op: {p.op}")

    sig = OP_SIGNATURES[p.op]
    if p.kind != sig["kind"]:
        raise TypeErrorKFQL(f"Op {p.op} expected kind={sig['kind']}, got {p.kind}")
    if len(p.args) < sig["arity"]:
        raise TypeErrorKFQL(f"Op {p.op} expected {sig['arity']} args, got {p.args}")


"""
TODO:
need to add more type checking for the IR, for evolutionary schema and user-defined operators (can give feedback to llms)
- temporal_always
- temporal_interframe
- returns
- query_name
- objects
- keyframes
- select
- dataset
"""
def type_check_ir(ir: IRRoot) -> None:
    """Semantic validation of IRRoot"""
    env_objects = {o.name: o for o in ir.objects}

    for kf in ir.keyframes:
        for p in kf.predicates:
            _check_predicate(p)

            # validate object references
            for a in p.args:
                if isinstance(a, str) and a not in env_objects:
                    raise SchemaError(f"Unknown entity reference: {a}")