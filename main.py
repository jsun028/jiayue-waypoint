import pickle
import argparse
import json
from pathlib import Path
import pandas as pd
from loguru import logger

from NL.registry import UDFRegistry
from NL.compiler import QueryCompiler
from NL.utils.nuscene_traj_viz import plot_bev_snapshot

from NL.utils.viz import _write_results_json, _generate_visualizations
from NL.specs import print_spec_details, QuerySpec


logger.add("runs.log", rotation="1 week")

# Example usage
def example_usage(spec_path, data_path, coverage: float | None = None, track_stats: bool = True, 
    out_path: str | None = None, do_viz: bool = False, viz_dir: str | None = None, limit: int | None = None, 
    dedup_threshold: float | None = None, metadata_path: str | None = None, estimation_mode: bool = False):

    # Sample data
    df = pd.read_csv(data_path)
    
    # UDF registry
    registry = UDFRegistry(df)

    # Load a sample spec (support both pkl and JSON)
    spec_path_obj = Path(spec_path)
    if spec_path_obj.suffix.lower() == ".json":
        # Load from JSON
        with open(spec_path, "r") as f:
            spec_dict = json.load(f)
        spec = QuerySpec.model_validate(spec_dict)
        print(f"Loaded spec from JSON: {spec_path}")
    else:
        # Load from pickle (default)
        with open(spec_path, "rb") as f:
            spec = pickle.load(f)
        print(f"Loaded spec from pickle: {spec_path}")
    
    print("spec: ", spec)
    print_spec_details(spec)

    print("[INFO] QueryCompiler initialized successfully with two-stage search implementation")
    print(f"Available UDFs: {list(registry.get_all_udfs().keys())}")
    
    # def __init__(self, registry: UDFRegistry, df: pd.DataFrame, logger: logger = None, coverage: float | None = None, track_stats: bool = True, dedup_threshold: float = 0.25, limit: int | None = None,
    compiler = QueryCompiler(
        registry=registry, df=df, logger=logger, 
        coverage=coverage, track_stats=track_stats, dedup_threshold=dedup_threshold, limit=limit, 
        metadata_path=metadata_path)
    # Execute query with two-stage search
    results = compiler.execute_query(spec, estimation_mode=estimation_mode)
    for result in results:
        print(result)

    # Optionally write results to JSON
    if out_path:
        _write_results_json(results, out_path)

    # Optionally generate visualizations per result and keyframe
    if do_viz:
        out_dir = Path(viz_dir) if viz_dir else Path(out_path).with_suffix("").with_name(Path(out_path).stem + "_viz") if out_path else Path("viz_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        _generate_visualizations(df, results, out_dir, top_k=limit)
    

if __name__ == "__main__":
    # example_usage()

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--coverage", type=float, default=1.0, help="Fraction of frames to scan (0-1]")
    parser.add_argument("--track-stats", action="store_true", help="Enable predicate selectivity stats")
    parser.add_argument("--out", type=str, default=None, help="Path to write results JSON")
    parser.add_argument("--viz", action="store_false", help="Generate visualization images per result keyframe")
    parser.add_argument("--viz-dir", type=str, default=None, help="Directory for visualization images (default derived from --out)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of results to return and visualize (top-k by score)")
    parser.add_argument("--dedup-threshold", type=float, default=0.25, help="Deduplication threshold for overlapping time windows")
    parser.add_argument("--metadata-path", type=str, default=None, help="Path to metadata JSON file")
    parser.add_argument("--estimation-mode", action="store_true", help="Enable estimation mode")
    args = parser.parse_args()

    logger.info(f"Running example usage with spec: {args.spec} and data: {args.data}, coverage={args.coverage}, track_stats={args.track_stats}, out={args.out}, viz={args.viz}")
    example_usage(
        args.spec,
        args.data,
        coverage=args.coverage,
        track_stats=args.track_stats,
        out_path=args.out,
        do_viz=args.viz,
        viz_dir=args.viz_dir,
        limit=args.limit,
        dedup_threshold=args.dedup_threshold,
        metadata_path=args.metadata_path,
        estimation_mode=args.estimation_mode,
    )

