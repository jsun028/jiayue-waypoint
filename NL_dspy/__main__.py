"""CLI entry point for running the DSPy NL → QuerySpec pipeline."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keyframeql.specs import print_spec_details
from keyframeql.utils.io import find_data_files
from stats_prompt import format_stats_for_prompt
from loguru import logger
logger.add("nl_dspy_runs.log", rotation="1 week")

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

if __package__ in {None, ""}:
    pkg_root = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(pkg_root.parent))
    from NL_dspy.pipeline import build_pipeline, configure_lm, run_pipeline, write_spec_pickle # type: ignore
else:
    from .pipeline import build_pipeline, configure_lm, run_pipeline, write_spec_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nl", help="Natural language request to translate.")
    parser.add_argument(
        "--model",
        default=os.getenv("DSPY_LM_MODEL", "openai/gpt-4o-mini"),
        help="Model identifier compatible with dspy.LM / LiteLLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("DSPY_LM_TEMPERATURE", "1.0")),
        help="Sampling temperature for the LM.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key (falls back to OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--dump-json",
        default=None,
        help="Optional path to write the raw spec JSON.",
    )
    parser.add_argument(
        "--dump-pickle",
        default=None,
        help="Optional path to write the QuerySpec pickle (compatible with NL/main.py).",
    )
    stats = parser.add_argument_group("dataset statistics")
    stats.add_argument(
        "--generate-stats",
        action="store_true",
        help="Compute statistics in-memory and include in the prompt.",
    )
    stats.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset CSV file for statistics generation.",
    )
    stats.add_argument(
        "--stats-sample",
        type=float,
        default=0.2,
        help="Sample ratio in (0,1] when generating stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Running DSPy pipeline with args: {args}")

    print("[DSPy][CLI] Initializing language model …")
    configure_lm(args.model, temperature=args.temperature, api_key=args.api_key)
    print("[DSPy][CLI] Building pipeline …")
    pipeline = build_pipeline(temperature=args.temperature)

    # Prepare optional stats text
    stats_text = None
    if args.generate_stats:

        # Resolve dataset CSV path via env or a common default
        dataset_dir = Path(args.dataset_dir).resolve()
        csv_list = find_data_files(dataset_dir, "*.csv", False, None)

        from keyframeql.optimizer.statistics_builder import KeyframeQLStatisticsBuilder  
        builder = KeyframeQLStatisticsBuilder(
            csv_list if len(csv_list) > 1 else csv_list[0],
            bins=20,
            sample_ratio=float(args.stats_sample),
            ego_bins=8,
        ).load_dataset().compute_statistics()
        stats_meta = builder.metadata

        stats_text = format_stats_for_prompt(stats_meta)
        print("[DSPy][CLI] Generated dataset statistics for prompt grounding")

    print("[DSPy][CLI] Running pipeline …")
    result = run_pipeline(args.nl, pipeline, stats_text=stats_text)

    print("=== QuerySpec (pydantic) ===")
    print(result.spec)

    logger.info(f"QuerySpec: {result.spec}")


    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as handle:
            handle.write(result.spec_json)
        print(f"\nRaw JSON saved to {args.dump_json}")

    logger.info(f"Raw JSON: {json.dumps(json.loads(result.spec_json), indent=2)}")

    print_spec_details(result.spec)

    if args.dump_pickle:
        write_spec_pickle(result.spec, args.dump_pickle)


if __name__ == "__main__":
    main()


