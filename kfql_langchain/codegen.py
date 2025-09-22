from ir import *
from collections import Counter

def _emit_objects(objs: List[IRObject]) -> str:
    lines = ["# ── object proxy ──────────────────────────────────────────────────────"]
    for o in objs:
        lines.append(f"{o.name} = Obj('{o.cls}', idx={o.idx})")
    return "\n".join(lines)

def _emit_keyframes(kfs: List[IRKeyframe]) -> str:
    lines = ["", "# ── keyframes ----------------------------------------------------------"]
    for k in kfs:
        preds = " & ".join(_emit_pred(p) for p in k.predicates)
        lines.append(f"{k.name} = KF('{k.name}').where({preds})")
    return "\n".join(lines)

def _emit_pred(p: IRPredicate) -> str:
    # single-frame
    if p.op == "velocity_above":
        return f"{p.args[0]}.velocity_above({float(p.args[1])})"
    if p.op == "velocity_below":
        return f"{p.args[0]}.velocity_below({float(p.args[1])})"
    if p.op == "heading_diff_to":
        return f"{p.args[0]}.heading_diff_to({p.args[1]}, {float(p.args[2])})"
    if p.op == "distance_within":
        return f"{p.args[0]}.distance_within({float(p.args[1])})"
    if p.op == "distance_apart":
        return f"{p.args[0]}.distance_apart({float(p.args[1])})"
    # inter-frame (comparator)
    if p.op == "heading_diff":
        return f"{p.args[0]}.heading_diff({float(p.args[1])})"
    raise ValueError(f"Unknown predicate: {p.op}")

def _emit_temporal(ir: IRRoot) -> str:
    lines = ["", "# ── build query --------------------------------------------------------", f"{ir.query_name} = ("]
    lines.append("    Query(" + ", ".join(k.name for k in ir.keyframes) + ")")
    # always
    for alw in ir.temporal_always:
        anchor_str = "None" if alw.anchor is None else alw.anchor  # ✅ anchor 반영
        lines.append(
            f"      .always(anchor={anchor_str}, dur_sec={float(alw.dur_sec)}, "
            f"target={alw.target}, tol={float(alw.tol)})"
        )
    # interframe
    for itf in ir.temporal_interframe:
        comps = ", ".join(_emit_pred(c) for c in itf.comparators)
        lines.append(
            "      .interframe("
            f"anchor={itf.anchor}, target={itf.target}, time_shift={float(itf.time_shift)}, "
            f"comparators=[{comps}])"
        )
    # count objects by class
    counts = Counter(o.cls for o in ir.objects)
    obj_map = ", ".join(f"'{cls}': {cnt}" for cls, cnt in counts.items())
    lines.append(f"      .build(objects={{ {obj_map} }})")
    lines.append(")")
    return "\n".join(lines)

def compile_to_kfql(ir: IRRoot) -> str:
    return "\n".join([
        _emit_objects(ir.objects),
        _emit_keyframes(ir.keyframes),
        _emit_temporal(ir),
    ])
