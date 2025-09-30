"""CLI entry point for running the DSPy NL → QuerySpec pipeline."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

if __package__ in {None, ""}:
    pkg_root = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(pkg_root.parent))
    from NL_dspy.pipeline import build_pipeline, configure_lm, run_pipeline  # type: ignore
else:
    from .pipeline import build_pipeline, configure_lm, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nl", help="Natural language request to translate.")
    parser.add_argument(
        "--model",
        default=os.getenv("DSPY_LM_MODEL", "openai/gpt-4o-mini"),
        help="Model identifier compatible with dspy.LM / LiteLLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("DSPY_LM_TEMPERATURE", "0.0")),
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[DSPy][CLI] Initializing language model …")
    configure_lm(args.model, temperature=args.temperature, api_key=args.api_key)
    print("[DSPy][CLI] Building pipeline …")
    pipeline = build_pipeline(temperature=args.temperature)

    print("[DSPy][CLI] Running pipeline …")
    result = run_pipeline(args.nl, pipeline)

    print("=== QuerySpec (pydantic) ===")
    print(result.spec)

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as handle:
            handle.write(result.spec_json)
        print(f"\nRaw JSON saved to {args.dump_json}")
    else:
        print("\n=== Raw JSON ===")
        print(json.dumps(json.loads(result.spec_json), indent=2))


if __name__ == "__main__":
    main()


