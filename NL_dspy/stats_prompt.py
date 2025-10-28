from __future__ import annotations

from typing import Any, Optional


def _quantile_from_hist(hist: dict, q: float) -> Optional[float]:
    """Best-effort percentile estimate from our quantile histogram format.

    The builder emits edges based on linspace quantiles. We can index into edges
    to approximate a desired percentile without raw data.
    """
    try:
        edges: list[float] = list(hist.get("bins", []))
        if not edges:
            return None
        # edges length = bins+1; quantiles are linear in index
        idx = int(round(q * (len(edges) - 1)))
        idx = max(0, min(idx, len(edges) - 1))
        return float(edges[idx])
    except Exception:  # noqa: BLE001
        return None


def format_stats_for_prompt(metadata: dict[str, Any], *, max_classes: int = 6) -> str:
    """Render a compact, LM-friendly guidance block from dataset statistics.

    Expected schema matches `KeyframeQLStatisticsBuilder.metadata`.
    """
    lines: list[str] = []
    ds_name = metadata.get("dataset_name")
    if ds_name:
        lines.append(f"Dataset: {ds_name}")
    n_frames = metadata.get("n_frames")
    n_objects = metadata.get("n_objects")
    if n_frames or n_objects:
        parts = []
        if isinstance(n_frames, int):
            parts.append(f"frames={n_frames}")
        if isinstance(n_objects, int):
            parts.append(f"objects={n_objects}")
        if parts:
            lines.append("Size: " + ", ".join(parts))

    # Attribute ranges and anchors
    attr_hist = metadata.get("attribute_histograms", {}) or {}
    def _fmt_hist_line(hist: Optional[dict], label: str) -> Optional[str]:
        if not hist:
            return None
        bins = hist.get("bins", [])
        if not bins:
            return None
        vmin = float(bins[0])
        vmax = float(bins[-1])
        p50 = _quantile_from_hist(hist, 0.50)
        p75 = _quantile_from_hist(hist, 0.75)
        p90 = _quantile_from_hist(hist, 0.90)
        stats_parts = [f"range={vmin:.3g}..{vmax:.3g}"]
        if p50 is not None:
            stats_parts.append(f"p50={p50:.3g}")
        if p75 is not None:
            stats_parts.append(f"p75={p75:.3g}")
        if p90 is not None:
            stats_parts.append(f"p90={p90:.3g}")
        return f"- {label}: " + ", ".join(stats_parts)

    def fmt_attr(name: str, label: str) -> Optional[str]:
        return _fmt_hist_line(attr_hist.get(name), label)

    attr_lines: list[str] = []
    maybe = [
        ("velocity_mag", "velocity_mag"),
        ("bbox_area", "bbox_area"),
        ("yaw", "yaw (rad)")
    ]
    for key, label in maybe:
        s = fmt_attr(key, label)
        if s:
            attr_lines.append(s)
    if attr_lines:
        lines.append("Attributes:")
        lines.extend(attr_lines)

    # Class distribution (top-k)
    class_dist = metadata.get("class_distribution", {}) or {}
    if isinstance(class_dist, dict) and class_dist:
        # Flatten into list of (class, ratio)
        items: list[tuple[str, float]] = []
        for cname, row in class_dist.items():
            try:
                items.append((str(cname), float(row.get("ratio", 0.0))))
            except Exception:  # noqa: BLE001
                continue
        items.sort(key=lambda x: x[1], reverse=True)
        if items:
            lines.append("Top classes (ratio):")
            for cname, ratio in items[:max_classes]:
                lines.append(f"- {cname}: {ratio:.2%}")

        # Per-class attribute stats for top classes
        class_attr = metadata.get("class_attribute_histograms", {}) or {}
        if isinstance(class_attr, dict) and class_attr and items:
            per_class_lines: list[str] = []
            maybe = [
                ("velocity_mag", "velocity_mag"),
                ("bbox_area", "bbox_area"),
                ("yaw", "yaw (rad)")
            ]
            for cname, _ in items[:max_classes]:
                hists = class_attr.get(cname) or {}
                if not hists:
                    continue
                class_lines: list[str] = []
                for key, label in maybe:
                    s = _fmt_hist_line(hists.get(key), label)
                    if s:
                        class_lines.append("  " + s)
                if class_lines:
                    per_class_lines.append(f"- {cname}:")
                    per_class_lines.extend(class_lines)
            if per_class_lines:
                lines.append("Per-class attributes:")
                lines.extend(per_class_lines)

    # Heuristic anchors for common predicates
    vel_hist = attr_hist.get("velocity_mag")
    vel_p60 = _quantile_from_hist(vel_hist, 0.60) if vel_hist else None
    vel_p80 = _quantile_from_hist(vel_hist, 0.80) if vel_hist else None
    vel_p30 = _quantile_from_hist(vel_hist, 0.30) if vel_hist else None

    bbox_hist = attr_hist.get("bbox_area")
    bbox_p25 = _quantile_from_hist(bbox_hist, 0.25) if bbox_hist else None
    bbox_p75 = _quantile_from_hist(bbox_hist, 0.75) if bbox_hist else None

    anchors: list[str] = []
    if vel_p60 is not None and vel_p80 is not None and vel_p30 is not None:
        anchors.append(
            f"- velocity_above: moving≈{vel_p60:.3g}–{vel_p80:.3g}; stopped < {vel_p30:.3g}"
        )
    if bbox_p75 is not None and bbox_p25 is not None:
        anchors.append(
            f"- bbox_area: large ≥ {bbox_p75:.3g}; small ≤ {bbox_p25:.3g}"
        )
    if anchors:
        lines.append("Heuristic anchors:")
        lines.extend(anchors)

    # Ego summary (optional)
    ego = metadata.get("ego") or {}
    if isinstance(ego, dict) and "summary" in ego:
        try:
            n_e = int(ego["summary"].get("n_frames", 0))
            lines.append(f"Ego frames: {n_e}")
        except Exception:  # noqa: BLE001
            pass

    # Pairwise distance guidance (optional)
    pair_hist = metadata.get("pairwise_distance_histogram") or {}
    if isinstance(pair_hist, dict) and pair_hist.get("bins"):
        p5 = _quantile_from_hist(pair_hist, 0.05)
        p25 = _quantile_from_hist(pair_hist, 0.25)
        p50 = _quantile_from_hist(pair_hist, 0.50)
        p75 = _quantile_from_hist(pair_hist, 0.75)
        lines.append("Pairwise distance (approximate):")
        parts: list[str] = []
        if p5 is not None:
            parts.append(f"p5={p5:.3g}")
        if p25 is not None:
            parts.append(f"p25={p25:.3g}")
        if p50 is not None:
            parts.append(f"median={p50:.3g}")
        if p75 is not None:
            parts.append(f"p75={p75:.3g}")
        if parts:
            lines.append("- " + ", ".join(parts))

    # Ego-to-agent distance guidance (optional)
    ego_pair = metadata.get("ego_to_agent_distance_histogram") or {}
    if isinstance(ego_pair, dict) and ego_pair.get("bins"):
        p5 = _quantile_from_hist(ego_pair, 0.05)
        p25 = _quantile_from_hist(ego_pair, 0.25)
        p50 = _quantile_from_hist(ego_pair, 0.50)
        p75 = _quantile_from_hist(ego_pair, 0.75)
        lines.append("Ego-to-agent distance (approximate):")
        parts: list[str] = []
        if p5 is not None:
            parts.append(f"p5={p5:.3g}")
        if p25 is not None:
            parts.append(f"p25={p25:.3g}")
        if p50 is not None:
            parts.append(f"median={p50:.3g}")
        if p75 is not None:
            parts.append(f"p75={p75:.3g}")
        if parts:
            lines.append("- " + ", ".join(parts))

    # Final guidance to steer the LM
    last_lines = [
        "Hints:",
        "-Use these ranges to choose realistic predicate values when NL is vague.",
    ]
    lines.extend(last_lines)
    return "\n".join(lines)


