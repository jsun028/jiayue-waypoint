import pickle
import argparse
import json
from pathlib import Path
import pandas as pd
from loguru import logger
from typing import List, Tuple

from keyframeql.registry import UDFRegistry
from keyframeql.compiler import QueryCompiler
from keyframeql.learner import Reranker
from keyframeql.utils.io import find_data_files
from keyframeql.specs import print_spec_details, QuerySpec
from dataset_specific.nuscene.viz import _generate_visualizations


logger.add("runs.log", rotation="1 week")


def search_loop(spec_path, data_files: List[Path], coverage: float | None = None, track_stats: bool = True, 
    limit: int | None = None, dedup_threshold: float | None = None, metadata_path: str | None = None, 
    estimation_mode: bool = False,
    slider_setting: str = "medium", dataset : str = "nuscene"):

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

    # Sample data
    all_results = []
    for idx, data_path in enumerate(data_files):
        df = pd.read_csv(data_path)
        
        # UDF registry
        registry = UDFRegistry(df)

        print("[INFO] QueryCompiler initialized successfully with two-stage search implementation")
        print(f"Available UDFs: {list(registry.get_all_udfs().keys())}")
        print(f"[INFO] Using slider setting: {slider_setting}")
        
        # def __init__(self, registry: UDFRegistry, df: pd.DataFrame, logger: logger = None, coverage: float | None = None, track_stats: bool = True, dedup_threshold: float = 0.25, limit: int | None = None,
        compiler = QueryCompiler(
            registry=registry, df=df, logger=logger, 
            coverage=coverage, track_stats=track_stats, dedup_threshold=dedup_threshold, limit=limit, 
            metadata_path=metadata_path, slider_setting=slider_setting,
            dataset=dataset)
        # Execute query with two-stage search
        results = compiler.execute_query(spec, estimation_mode=estimation_mode)
        for result in results:
            all_results.append((idx, result))
    
   # Sort by aggregate_score
    sorted_results = sorted(all_results, 
                         key=lambda x: x[1]['aggregate_score'], 
                         reverse=True)
        
    return sorted_results

def plot_top_k(ranked_results: List[Tuple], data_files: List[str], 
               viz_dir: str, k=10):
    out_dir = Path(viz_dir) 
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(k):
        dataset_idx = ranked_results[i][0]
        df = pd.read_csv(data_files[dataset_idx])
        _generate_visualizations(df, [ranked_results[i][1]], out_dir, top_k=1, fname=f"{i+1}")


def active_learning_loop(top_results, data_files, viz_dir):
    reranker = Reranker(top_results)
    reranker.label_and_learn(top_results, data_files, viz_dir)
    ranked_results = reranker.rerank_results(top_results)
    return ranked_results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True, help="Folder containing data files (e.g., CSVs)")
    parser.add_argument("--dataset", type=str, default='nuscene', required=True, help="Name of dataset (nuscene, virat...)")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern to match data files")
    parser.add_argument("--coverage", type=float, default=1.0, help="Fraction of frames to scan (0-1]")
    parser.add_argument("--track-stats", action="store_true", help="Enable predicate selectivity stats")
    parser.add_argument("--out", type=str, default=None, help="Path to write results JSON")
    parser.add_argument("--viz-dir", type=str, default=None, help="Directory for visualization images (default derived from --out)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of results to return and visualize (top-k by score)")
    parser.add_argument("--dedup-threshold", type=float, default=0.25, help="Deduplication threshold for overlapping time windows")
    parser.add_argument("--metadata-path", type=str, default=None, help="Path to metadata JSON file")
    parser.add_argument("--estimation-mode", action="store_true", help="Enable estimation mode")
    parser.add_argument("--slider-setting", type=str, default="medium", choices=["low", "medium", "high"],
                       help="Slider setting for DiscreteSlider values: 'low' (most selective), 'medium' (balanced), 'high' (most permissive)")
    args = parser.parse_args()


    dataset_dir = Path(args.dataset_dir).resolve()
    data_files = find_data_files(dataset_dir, args.pattern, False, 10)
    if not data_files:
        print("No data files found. Nothing to do.")
        exit(1)

    logger.info(f"Running example usage with spec: {args.spec} and data: {args.dataset_dir}, coverage={args.coverage}, track_stats={args.track_stats}, out={args.out}, slider_setting={args.slider_setting}")
    top_results = search_loop(
        args.spec,
        data_files,
        coverage=args.coverage,
        track_stats=args.track_stats,
        limit=args.limit,
        dedup_threshold=args.dedup_threshold,
        metadata_path=args.metadata_path,
        estimation_mode=args.estimation_mode,
        slider_setting=args.slider_setting,
        dataset=args.dataset
    )
    
    # Plot top-k before reranking
    plot_top_k(top_results, data_files, "viz_out/raw/", 5)
    with open('viz_out/raw/results.pkl', 'wb') as f:
        pickle.dump(top_results, f)

    # with open('viz_out/raw/results.pkl', 'rb') as f:
    #     top_results = pickle.load(f)
    # ranked_results = active_learning_loop(top_results, data_files, args.viz_dir)
    # plot_top_k(ranked_results, data_files, "viz_out/reranked/", 5)
    # with open('viz_out/reranked/results.pkl', 'wb') as f:
    #     pickle.dump(ranked_results, f)
