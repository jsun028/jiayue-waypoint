from langchain.chains.base import Chain
from typing import Dict, Any

from ir import IRRoot, IRObject, IRKeyframe, IRPredicate, IRTemporalAlways, IRTemporalInterframe
from checker import type_check_ir
from codegen import compile_to_kfql

class IRTransformChain(Chain):
    """Chain: JSON IR (dict) → IRRoot → DSL string"""

    @property
    def input_keys(self):
        return ["ir_json"]

    @property
    def output_keys(self):
        return ["dsl"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ir_json = inputs["ir_json"]

        if hasattr(ir_json, "dict"):
            ir_json = ir_json.dict()
        
        # if ir_json is string, convert to dict
        if isinstance(ir_json, str):
            import json
            ir_json = json.loads(ir_json)

        # Dict → IRRoot
        objs = [IRObject(**o) for o in ir_json.get("objects", [])]
        kfs = [IRKeyframe(
            name=k["name"],
            predicates=[IRPredicate(**p) for p in k.get("predicates", [])]
        ) for k in ir_json.get("keyframes", [])]
        alws = [IRTemporalAlways(**a) for a in ir_json.get("temporal_always", [])]
        itfs = [IRTemporalInterframe(
            anchor=i["anchor"], target=i["target"],
            time_shift=i["time_shift"],
            comparators=[IRPredicate(**p) for p in i.get("comparators", [])]
        ) for i in ir_json.get("temporal_interframe", [])]

        ir = IRRoot(
            select=ir_json.get("select", "event"),
            dataset=ir_json.get("dataset", "TrafficVideo"),
            objects=objs,
            keyframes=kfs,
            temporal_always=alws,
            temporal_interframe=itfs,
            returns=ir_json.get("returns", []),
            query_name=ir_json.get("query_name", "AUTO_GEN_QUERY"),
        )

        # type check
        type_check_ir(ir)

        # DSL generation
        dsl = compile_to_kfql(ir)
        return {"dsl": dsl}
