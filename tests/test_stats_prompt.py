"""Example: compose a SpecGenerator prompt with dataset statistics included.

Run directly:
  python test_stats_prompt.py --stats-json metadata/scene_scene-0225_stats.json
or compute from CSV on the fly:
  python test_stats_prompt.py --stats-csv path/to/dataset.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keyframeql.registry import GLOBAL_UDF_REGISTRY
from NL_dspy.pipeline import SpecGenerator, _format_available_udfs_for_prompt
from NL_dspy.stats_prompt import format_stats_for_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stats-json",
        default=str(Path("metadata") / "scene_scene-0225_stats.json"),
        help="Path to precomputed stats JSON (default points at repository sample).",
    )
    p.add_argument(
        "--stats-csv",
        default=None,
        help="Optional CSV to compute stats if JSON not available.",
    )
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--ego-bins", type=int, default=8)
    p.add_argument("--sample", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load or compute metadata
    metadata = None
    if args.stats_json and Path(args.stats_json).exists():
        with open(args.stats_json, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    elif args.stats_csv:
        from keyframeql.optimizer.statistics_builder import KeyframeQLStatisticsBuilder  # type: ignore

        builder = KeyframeQLStatisticsBuilder(
            args.stats_csv,
            bins=args.bins,
            sample_ratio=args.sample,
            ego_bins=args.ego_bins,
        ).load_dataset().compute_statistics()
        metadata = builder.metadata
    else:
        print("[Example] No stats provided and default JSON not found; continuing without stats.")

    stats_text = format_stats_for_prompt(metadata) if metadata else None

    # Prepare UDF list
    available = _format_available_udfs_for_prompt(GLOBAL_UDF_REGISTRY.get_all_udfs().keys())

    # Compose prompt including stats
    generator = SpecGenerator(verbose=False)
    nl = "A car moving fast, then slowing down and making a right turn."
    prompt = generator._compose_prompt(nl, available, stats_text=stats_text)

    print("=" * 80)
    print("FULL PROMPT WITH DATASET STATS")
    print("=" * 80)
    print(prompt)
    print("=" * 80)


if __name__ == "__main__":
    main()


