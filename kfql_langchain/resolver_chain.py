from langchain.chains.base import Chain
from typing import Dict, Any
from udf_resolver import UDFResolver

class UDFResolveChain(Chain):
    """Chain: IR JSON → UDFResolver → cleaned IR JSON"""

    def __init__(self, resolver: UDFResolver):
        super().__init__()
        self.resolver = resolver

    @property
    def input_keys(self):
        return ["ir_json"]

    @property
    def output_keys(self):
        return ["ir_json"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ir_json = inputs["ir_json"]

        # Predicates 내부에서 UDF 이름 확인
        for kf in ir_json.get("keyframes", []):
            for pred in kf.get("predicates", []):
                op = pred["op"]
                decision = self.resolver.resolve(op)
                # 필요하면 op를 수정하거나 로그만 남김
                print(f"[UDFResolver] op={op} → {decision}")

        return {"ir_json": ir_json}